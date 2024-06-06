"""
Solve the Orr-Sommerfeld eigenvalue problem

"""
import warnings
from scipy.linalg import eig
#from numpy.linalg import eig
#from numpy.linalg import inv
import numpy as np
import sympy as sp
from shenfun import FunctionSpace, Function, Dx, inner, TestFunction, \
    TrialFunction

np.seterr(divide='ignore')

#pylint: disable=no-member

try:
    from matplotlib import pyplot as plt

except ImportError:
    warnings.warn("matplotlib not installed")

x = sp.Symbol('x', real=True)

class OrrSommerfeld_eigs:
    def __init__(self, alfa=1., Re=8000., N=80, K0=0, K1=0, **kwargs):
        kwargs.update(dict(alfa=alfa, Re=Re, N=N, K0=K0, K1=K1))
        vars(self).update(kwargs)
        self.x, self.w = None, None

    def interp(self, y, eigvals, eigvectors, eigval=1, verbose=False):
        """Interpolate solution eigenvector and it's derivative onto y

        Parameters
        ----------
            y : array
                Interpolation points
            eigvals : array
                All computed eigenvalues
            eigvectors : array
                All computed eigenvectors
            eigval : int, optional
                The chosen eigenvalue, ranked with descending imaginary
                part. The largest imaginary part is 1, the second
                largest is 2, etc.
            verbose : bool, optional
                Print information or not
        """
        nx, eigval = self.get_eigval(eigval, eigvals, verbose)
        SB = self.get_trialspace(dtype='D')
        phi_hat = Function(SB)
        phi_hat[:-4] = np.squeeze(eigvectors[:, nx])
        phi = phi_hat.eval(y)
        dphidy = Dx(phi_hat, 0, 1).eval(y)
        return eigval, phi, dphidy

    def get_trialspace(self, dtype='d'):
        if (self.K0 == 0 and self.K1 == 0): # Regular channel
            return FunctionSpace(self.N, 'C', basis='ShenBiharmonic', dtype=dtype)
        if (self.K0 == 0 and self.K1 == np.inf):
            bcs = {'left': {'D': 0, 'N': 0},
                   'right': {'D': 0, 'N2': 0}}
        else:
            bcs = {'left': {'D': 0, 'W': (-self.K0, 0)},
                   'right': {'D': 0, 'W': (self.K1, 0)}}
        return FunctionSpace(self.N, 'C', bc=bcs, dtype=dtype) 
    
    def get_U(self):
        K0, K1 = self.K0, self.K1
        if K1 == np.inf:
            return 3 + 2*x - x**2
        return 1 + 2*(K0+K1+2*K0*K1 - (K0-K1)*x)/(2+K0+K1) - x**2

    def assemble(self, scale=None):
        V = self.get_trialspace(dtype='d')
        v = TestFunction(V)
        u = TrialFunction(V)

        Re = self.Re
        a = self.alfa
        g = 1j*a*Re
        U = self.get_U()

        K = inner(v, Dx(u, 0, 2))
        K1 = inner(v*U, u)
        K2 = inner(v*U, Dx(u, 0, 2))
        Q = inner(v, Dx(u, 0, 4))
        M = inner(v, u)

        B = -g*(K-a**2*M)
        A = Q-2*a**2*K+(a**4 - 2*g)*M - g*(K2-a**2*K1)
        A, B = A.diags().toarray(), B.diags().toarray()
        return A, B

    def solve(self, verbose=False, scale=None):
        """Solve the Orr-Sommerfeld eigenvalue problem
        """
        if verbose:
            print('Solving the Orr-Sommerfeld eigenvalue problem...')
            print('Re = '+str(self.Re)+' and alfa = '+str(self.alfa))
        A, B = self.assemble(scale=scale)
        return eig(A, B)

    @staticmethod
    def get_eigval(nx, eigvals, verbose=False):
        """Get the chosen eigenvalue

        Parameters
        ----------
            nx : int
                The chosen eigenvalue. nx=1 corresponds to the one with the
                largest imaginary part, nx=2 the second largest etc.
            eigvals : array
                Computed eigenvalues
            verbose : bool, optional
                Print the value of the chosen eigenvalue. Default is False.

        """
        indices = np.argsort(np.imag(eigvals))
        indi = indices[-1*np.array(nx)]
        eigval = eigvals[indi]
        if verbose:
            ev = list(eigval) if np.ndim(eigval) else [eigval]
            indi = list(indi) if np.ndim(indi) else [indi]
            for i, (e, v) in enumerate(zip(ev, indi)):
                print('Eigenvalue {} ({}) = {:2.16e}'.format(i+1, v, e))
        return indi, eigval

def check_arg(value):
    if np.isinf(eval(value)):
        return np.inf
    try:
        return int(value)
    except ValueError:
        try:
            u = tuple(map(int, value.strip('()').split(',')))
            return sp.Rational(*u)
        except ValueError:
            raise argparse.ArgumentTypeError(f'Invalid value: {value}')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Orr Sommerfeld parameters')
    parser.add_argument('--N', type=int, default=400,
                        help='Number of discretization points')
    parser.add_argument('--Re', default=8000.0, type=float,
                        help='Reynolds number')
    parser.add_argument('--alfa', default=1.0, type=float,
                        help='Parameter')
    parser.add_argument('--K0', default=0, type=check_arg,
                        help="Slip BC at x=-1. psi'-K0*psi''=0. K0=0 is no-slip. If a tuple, then it is assumed a rational number")
    parser.add_argument('--K1', default=0, type=check_arg,
                        help="Slip BC at x=1. psi'+K1*psi''=0. K1 = 0 is no-slip. If a tuple, then it is assumed a rational number") 
    parser.add_argument('--plot', dest='plot', action='store_true', help='Plot eigenvalues')
    parser.add_argument('--verbose', dest='verbose', action='store_true', help='Print results')
    parser.set_defaults(plot=False)
    parser.set_defaults(verbose=False)
    args = parser.parse_args()
    #z = OrrSommerfeld(N=120, Re=5772.2219, alfa=1.02056)
    z = OrrSommerfeld_eigs(**vars(args))
    evals, evectors = z.solve(verbose=args.verbose, scale=(0, -2))
    d = z.get_eigval(1, evals, args.verbose)

    if args.Re == 8000.0 and args.alfa == 1.0 and args.N > 80 and args.K0 == 0 and args.K1 == 0:
        assert abs(d[1] - (0.24707506017508621+0.0026644103710965817j)) < 1e-12

    if args.plot:
        plt.figure()
        evi = evals*z.alfa
        plt.plot(evi.imag, evi.real, 'o')
        plt.axis([-10, 0.1, 0, 1])
        plt.show()
