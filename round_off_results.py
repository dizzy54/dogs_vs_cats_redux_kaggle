with open('data/predictions.csv') as f:
    f.readline()
    predictions = f.read().splitlines()

with open('data/predictions_rounded.csv', 'w') as f:
    f.write('id,label\n')
    for pred in predictions:
        sno, p = pred.split(',')
        p_round = 0 if float(p) < 0.5 else 1
        f.write(str(sno) + ',' + '%.2f' % p_round)
        f.write('\n')
