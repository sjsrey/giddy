import giddy
import libpysal.api as lps
import libpysal
import mapclassify.api as mc
import numpy as np


f = lps.open(libpysal.examples.get_path("usjoin.csv"))
pci = np.array([f.by_col[str(y)] for y in range(1929,2010)])
q5 = np.array([mc.Quantiles(y).yb for y in pci]).transpose()
m = giddy.markov.Markov(q5)
print(m.transitions)
sqrt_n = 3
n = sqrt_n**2
w = lps.lat2W(sqrt_n, sqrt_n)
y = np.random.random_integers(4, size=(n, 50))
y = y - 1
m = giddy.markov.Markov(y)

# conditional
m_cond = giddy.markov.Geo_Markov(y, w, k=4, y_discrete=True)


