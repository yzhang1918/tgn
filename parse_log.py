import numpy as np
import re
import argparse



def run(name):
    total_time_pattern = r'.*Epoch +\d+.*total.* (\d*?.\d*)s$'
    train_time_pattern = r'.*Epoch +\d+.*training.* (\d*?.\d*)s$'
    old_vals_pattern = r'.*Test.*Old.*auc: (.+), ap: (.+)$'
    new_vals_pattern = r'.*Test.*New.*auc: (.+), ap: (.+)$'

    total_times = []
    train_times = []
    aps = []
    aucs = []
    ind_aps = []
    ind_aucs = []

    with open(name) as fh:
        lines = fh.read().strip().split('\n')
        head = lines[0]
        head_pattern = r".*data='(\w+)',.*prefix='([\w-]+)',.*"
        print(re.match(head_pattern, head).groups())
        for line in lines[1:]:
            r = re.match(total_time_pattern, line)
            if r is not None:
                total_times.append(float(r.groups()[0]))
                continue

            r = re.match(train_time_pattern, line)
            if r is not None:
                train_times.append(float(r.groups()[0]))
                continue

            r = re.match(old_vals_pattern, line)
            if r is not None:
                auc, ap = [float(v) for v in r.groups()]
                aucs.append(auc)
                aps.append(ap)
                continue

            r = re.match(new_vals_pattern, line)
            if r is not None:
                auc, ap = [float(v) for v in r.groups()]
                ind_aucs.append(auc)
                ind_aps.append(ap)
                continue

    print(f'# Times: {len(train_times)} # Evals: {len(aps)}')
    print(f'Train Time: {np.mean(train_times):.4f} std={np.std(train_times):.4f}')
    print(f'Total Time: {np.mean(total_times):.4f} std={np.std(total_times):.4f}')
    print(f'        AP: {np.mean(aps):.4f} std={np.std(aps):.4f}')
    print(f'   Ind  AP: {np.mean(ind_aps):.4f} std={np.std(ind_aps):.4f}')
    print(f'       AUC: {np.mean(aucs):.4f} std={np.std(aucs):.4f}')
    print(f'   Ind AUC: {np.mean(ind_aucs):.4f} std={np.std(ind_aucs):.4f}')




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('name', type=str)
    args = parser.parse_args()
    run(args.name)