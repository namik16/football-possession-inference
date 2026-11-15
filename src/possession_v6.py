"""Independent reliability-aware possession inference. No V4/V5 or GT inputs."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from games import GAME, GAMES, half_video_blocks
from starter.load_smart_tagging import load as load_tags

OUT = ROOT / 'output'
FPS = GAMES[GAME]['fps']
LENGTH, WIDTH = GAMES[GAME]['pitch']
BLOCKS = half_video_blocks(GAME)
N = GAMES[GAME]['n_video_frames']

# Physical priors, not fitted on public GT.
DIST_SCALE_M = 3.5
CONTROL_RADIUS_M = 7.0
MAX_BALL_STEP_M = 2.0  # between adjacent 25 Hz observations
EVIDENCE_DECAY = 0.96
SWITCH_THRESHOLD = 1.7
INITIAL_THRESHOLD = 0.5
MAX_EVIDENCE = 6.0
TAG_WEIGHT = 0.025  # weak relative to physical evidence
RESTART_PRE_S = 1.5
RESTART_GUARD_S = 0.2


def read_inputs():
    ball = pd.read_parquet(OUT / 'tracking_ball_graded.parquet').sort_values('frame').reset_index(drop=True)
    players = pd.read_parquet(OUT / 'tracking_players_graded.parquet')
    expected = np.concatenate([np.arange(first, last + 1) for _, first, last in BLOCKS])
    np.testing.assert_array_equal(ball.frame.to_numpy(), expected)
    if not ball.frame.is_unique:
        raise ValueError('Duplicate ball frames')
    if not set(players.team.dropna().unique()).issubset({'H', 'A'}):
        raise ValueError('Unknown player team')
    return ball, players


def physical_features(ball, players):
    p = players.merge(ball[['frame', 'ball_x', 'ball_y']], on='frame', how='left', validate='many_to_one')
    p['distance_m'] = np.hypot((p.x - p.ball_x) * LENGTH, (p.y - p.ball_y) * WIDTH)
    near = (p.groupby(['frame', 'team']).distance_m.min().unstack('team')
              .reindex(columns=['H', 'A']))
    result = ball[['frame', 'half', 'ball_x', 'ball_y', 'ball_speed', 'ball_visible']].copy()
    result = result.join(near, on='frame').rename(columns={'H': 'home_distance_m', 'A': 'away_distance_m'})
    h = result.home_distance_m.to_numpy(float)
    a = result.away_distance_m.to_numpy(float)
    visible = result.ball_visible.fillna(False).to_numpy(bool)
    result['home_score'] = np.where(visible & np.isfinite(h), np.exp(-h / DIST_SCALE_M), 0.0)
    result['away_score'] = np.where(visible & np.isfinite(a), np.exp(-a / DIST_SCALE_M), 0.0)
    # Check jumps only between consecutive frames within the same half.
    x = result.ball_x.to_numpy(float)
    y = result.ball_y.to_numpy(float)
    frames = result.frame.to_numpy(int)
    halves = result.half.to_numpy(int)
    jumps = np.zeros(len(result), bool)
    adjacent = (frames[1:] == frames[:-1] + 1) & (halves[1:] == halves[:-1])
    pair = visible[1:] & visible[:-1] & adjacent
    displacement = np.hypot(np.diff(x) * LENGTH, np.diff(y) * WIDTH)
    jumps[1:] = pair & (displacement > MAX_BALL_STEP_M)
    # A suspicious jump invalidates only the current observation.
    distance = np.fmin(h, a)
    reliability = np.where(visible, 1.0, 0.0)
    reliability[jumps] = 0.0
    reliability[~np.isfinite(distance)] = 0.0
    reliability[distance > CONTROL_RADIUS_M] *= 0.25
    result['ball_jump'] = jumps
    result['reliability'] = reliability
    result['physical_net'] = reliability * (result.home_score - result.away_score)
    return result


def smart_tag_features(result):
    """Map overlapping team tags to a bounded, weak contextual prior."""
    n = len(result)
    frames = result.frame.to_numpy(int)
    home = np.zeros(n, dtype=np.int16)
    away = np.zeros(n, dtype=np.int16)
    restart = np.zeros(n, bool)
    intervals = []
    for event in load_tags():
        code = str(event.get('code') or '').strip().lower()
        tags = event.get('tags') or {}
        team = str(tags.get('Team') or '').strip().lower()
        start, end = int(event['start_frame']), int(event['end_frame'])
        if end <= start:
            continue
        # Team-phase tags are weak corroboration, not direct possession labels.
        if code != 'set pieces' and code != 'goals':
            lo, hi = np.searchsorted(frames, [start, end])
            if team in {'barcelona', 'fc barcelona'}:
                home[lo:hi] += 1
            elif team in {'athletic club', 'athletic club de bilbao', 'athletic bilbao'}:
                away[lo:hi] += 1
        if code == 'set pieces':
            # Only a candidate interval; the tag endpoint is not a proven restart instant.
            stop = end - round(RESTART_GUARD_S * FPS)
            begin = max(start, stop - round(RESTART_PRE_S * FPS))
            if stop > begin:
                lo, hi = np.searchsorted(frames, [begin, stop])
                restart[lo:hi] = True
                if hi > lo:
                    intervals.append((begin, stop, str(tags.get('Type') or '')))
    # Conflicting simultaneous tags cancel, rather than double-count.
    context = np.sign(home.astype(int) - away.astype(int)).astype(float)
    result['tag_context'] = context
    result['tag_conflict'] = (home > 0) & (away > 0)
    result['restart_candidate'] = restart
    return result, pd.DataFrame(intervals, columns=['start_frame', 'end_frame_exclusive', 'type'])


def infer_half(group):
    group = group.copy()
    labels = []
    evidence_values = []
    reasons = []
    owner = 'X'
    evidence = 0.0
    prev_dead = False
    for row in group.itertuples(index=False):
        if row.restart_candidate:
            labels.append('D')
            evidence_values.append(0.0)
            reasons.append('restart_candidate')
            evidence = 0.0
            owner = 'X'
            prev_dead = True
            continue
        if prev_dead:
            # Restart ownership must be re-established from independent evidence.
            evidence = 0.0
            owner = 'X'
            prev_dead = False
        net = float(row.physical_net) + TAG_WEIGHT * float(row.tag_context)
        evidence = float(np.clip(EVIDENCE_DECAY * evidence + net, -MAX_EVIDENCE, MAX_EVIDENCE))
        if owner == 'X':
            if evidence >= INITIAL_THRESHOLD:
                owner = 'H'
            elif evidence <= -INITIAL_THRESHOLD:
                owner = 'A'
        elif owner == 'H' and evidence <= -SWITCH_THRESHOLD:
            owner = 'A'
        elif owner == 'A' and evidence >= SWITCH_THRESHOLD:
            owner = 'H'
        labels.append(owner)
        evidence_values.append(evidence)
        if row.reliability == 0:
            reasons.append('context_or_carry')
        elif row.ball_jump:
            reasons.append('ball_jump')
        else:
            reasons.append('physical_evidence')
    group['label'] = labels
    group['evidence_state'] = evidence_values
    group['reason'] = reasons
    return group


def build_spells(df):
    parts = []
    for half, g in df.groupby('half', sort=True):
        g = g.copy()
        g['spell_id'] = g.label.ne(g.label.shift()).cumsum()
        s = g.groupby('spell_id').agg(label=('label', 'first'), start_frame=('frame', 'first'),
                                       end_frame=('frame', 'last'), frames=('frame', 'size'),
                                       visible_frames=('ball_visible', 'sum'),
                                       mean_reliability=('reliability', 'mean')).reset_index(drop=True)
        s.insert(0, 'half', half)
        s['duration_s'] = s.frames / FPS
        s['ball_visibility_pct'] = 100 * s.visible_frames / s.frames
        parts.append(s)
    return pd.concat(parts, ignore_index=True)


def main():
    ball, players = read_inputs()
    features = physical_features(ball, players)
    features, restart_intervals = smart_tag_features(features)
    predictions = pd.concat([infer_half(g) for _, g in features.groupby('half', sort=True)], ignore_index=True)
    spells = build_spells(predictions)
    OUT.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(OUT / 'possession_v6_frames.parquet', index=False)
    spells.to_csv(OUT / 'possession_v6_spells.csv', index=False)
    restart_intervals.to_csv(OUT / 'possession_v6_restart_candidates.csv', index=False)
    labels = np.full(N, 'X', dtype='<U1')
    labels[predictions.frame.to_numpy(int)] = predictions.label.to_numpy()
    pd.DataFrame({'frame': np.arange(N), 'label': labels}).to_csv(OUT / 'possession_v6_submission.csv', index=False)
    possession = spells[spells.label.isin(['H', 'A'])]
    print('\n--- V6 DIAGNOSTICS ---')
    print('Graded frames:', len(predictions))
    print('Labels:\n', predictions.label.value_counts().to_string())
    print('Ball visibility:', round(100 * predictions.ball_visible.mean(), 2), '%')
    print('Ball jumps:', int(predictions.ball_jump.sum()))
    print('Low-reliability frames:', int((predictions.reliability < 0.5).sum()))
    print('Conflicting tag frames:', int(predictions.tag_conflict.sum()))
    print('Restart intervals:', len(restart_intervals))
    print('Possession spells:', len(possession))
    if len(possession):
        print('Median possession seconds:', round(possession.duration_s.median(), 2))
        print('Longest possession seconds:', round(possession.duration_s.max(), 2))
        print('Spells over 60 seconds:', int((possession.duration_s > 60).sum()))
        print(possession.nlargest(10, 'duration_s')[['half', 'label', 'start_frame', 'end_frame', 'duration_s', 'ball_visibility_pct']].to_string(index=False))
    print('Saved:', OUT / 'possession_v6_submission.csv')


if __name__ == '__main__':
    main()
