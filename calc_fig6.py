import sys
import numpy as np
import pandas as pd
import time
from scipy import stats
from permutations_stats.permutations import permutation_test

def calc_fig6(case):
  m_arr = np.arange(3,11)
  n_repeat = 10**4
  n_perm = 10**4
  np.random.seed(42)
  t = time.time()
  out_list = []
  for m in m_arr:
    print(m)
    if case == 1:
      n = m
    else:
      n = 2 * m
    out_list2 = []
    for _ in range(n_repeat):
      x_arr = np.random.randn(n)
      y_arr = np.random.randn(m)
      if case == 2:
        y_arr *= 2
      if case == 3:
        x_arr *= 2
      A = (y_arr - x_arr.reshape(-1,1) > 0).astype(int)
      q_hat = A.mean()
      sx2 = A.sum(axis=1).var(ddof=1)
      sy2 = A.sum(axis=0).var(ddof=1)
      if (sx2==0) and (sy2==0):
        sx2 = 1/n
        sy2 = 1/m
      s2 =  sx2 / (n * m**2) + sy2 / (n**2 * m)
      nu = (sx2/m + sy2/n)**2 / (sx2**2/((n-1)*m**2) + sy2**2/((m-1)*n**2))
      W = (q_hat - 0.5) / s2**0.5
      p_bm = 2 * stats.t(nu).sf(np.abs(W))
      p_perm = permutation_test(x_arr, y_arr, test='brunner_munzel',
                                method='simulation', n_iter=n_perm,
                                seed=np.random.randint(10**7))[1]
      out_list2.append(dict(p_bm=p_bm, p_perm=p_perm))
    df = pd.DataFrame(out_list2)
    sr = (df<=0.05).mean()
    sr['m'] = m
    out_list.append(sr)
  df = pd.concat(out_list, axis=1).T
  df['m'] = df.m.astype(int)
  df.to_csv('tmp.csv', index=False)
  print(f'{time.time()-t:.0f} seconds')

if __name__ == '__main__':
  case = int(sys.argv[1])  # 1, 2, or 3
  calc_fig6(case)
