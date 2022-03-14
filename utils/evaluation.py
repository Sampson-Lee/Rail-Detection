import numpy as np
import json
from scipy import special
from IPython import embed

color_list = [(0,0,225), (255,0,0), (0,225,0), (255,0,225), (255,255,225), (0,255,255), (255,255,0), (125,255,255)]
thickness_list = [1, 3, 5, 7, 9, 11, 13, 15]
thickness_list.reverse()

def grid_2_inter(out, griding_num):

    out = out.data.cpu().numpy()
    out_loc = np.argmax(out,axis=0)

    prob = special.softmax(out[:-1, :, :], axis=0)
    idx = np.arange(griding_num)
    idx = idx.reshape(-1, 1, 1)

    loc = np.sum(prob * idx, axis=0)

    loc[out_loc == griding_num] = griding_num
    out_loc = loc

    lanes = []
    for i in range(out_loc.shape[1]):
        out_i = out_loc[:,i]
        lane = [int(round((loc + 0.5) * 1280.0 / (griding_num - 1))) if loc != griding_num else -2 for loc in out_i]
        lanes.append(lane)
    return np.array(lanes)

def mask_2_inter(mask, row_anchor, num_lanes=4):

    all_idx = np.zeros((num_lanes, len(row_anchor)))

    for i, r in enumerate(row_anchor):
        label_r = np.asarray(mask)[int(round(r))]
        for lane_idx in range(1, num_lanes + 1):
            pos = np.where(label_r == lane_idx)[0]
            # pos = np.where(label_r == color_list[lane_idx])[0]
            if len(pos) == 0:
                all_idx[lane_idx - 1, i] = -1
                continue
            pos = np.mean(pos)
            all_idx[lane_idx - 1, i] = pos

    return all_idx


class LaneEval(object):
    pixel_thresh = 20
    pt_thresh = 0.85

    @staticmethod
    def line_accuracy(pred, gt, thresh):
        pred = np.array([p if p >= 0 else -100 for p in pred])
        gt = np.array([g if g >= 0 else -100 for g in gt])
        return np.sum(np.where(np.abs(pred - gt) < thresh, 1., 0.)) / len(gt)

    @staticmethod
    def bench(pred, gt, y_samples):
        # embed()
        if any(len(p) != len(y_samples) for p in pred):
            raise Exception('Format of lanes error.')

        line_accs = []
        fp, fn = 0., 0.
        matched = 0.
        for x_gts, x_preds in zip(gt, pred):
            acc = LaneEval.line_accuracy(np.array(x_preds), np.array(x_gts), LaneEval.pixel_thresh)
            if acc < LaneEval.pt_thresh:
                fn += 1
            else:
                matched += 1
            line_accs.append(acc)
        fp = len(pred) - matched
        if len(gt) > 4 and fn > 0:
            fn -= 1
        s = sum(line_accs)
        if len(gt) > 4:
            s -= min(line_accs)
        return s / max(min(4.0, len(gt)), 1.), fp / len(pred) if len(pred) > 0 else 0., fn / max(min(len(gt), 4.) , 1.)

    @staticmethod
    def bench_all(preds, gts, y_samples):
        accuracy, fp, fn = 0., 0., 0.
        for pred, gt in zip(preds, gts):
            try:
                a, p, n = LaneEval.bench(pred, gt, y_samples)
            except BaseException as e:
                raise Exception('Format of lanes error.')
            accuracy += a
            fp += p
            fn += n
        num = len(gts)
        # the first return parameter is the default ranking parameter
        return json.dumps([
            {'name': 'Accuracy', 'value': accuracy / num, 'order': 'desc'},
            {'name': 'FP', 'value': fp / num, 'order': 'asc'},
            {'name': 'FN', 'value': fn / num, 'order': 'asc'}
        ])

if __name__ == '__main__':
    from data.constant import raildb_row_anchor
    preds = np.random.randint(0, high=1280, size=(4, 4, len(raildb_row_anchor)))
    gts = np.random.randint(0, high=1280, size=(4, 4, len(raildb_row_anchor)))
    res = LaneEval.bench_all(preds, gts, raildb_row_anchor)
    res = json.loads(res)
    for r in res:
        print(r['name'], r['value'])