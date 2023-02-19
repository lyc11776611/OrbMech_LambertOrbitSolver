from Settings import *


def S(z):

    if z > 0:
        return (z**0.5-np.sin(z**0.5))/(z**0.5)**3

    elif z==0:
        return 1/6.0

    else:
        return (np.sinh((-z)**0.5)-(-z)**0.5)/((-z)**0.5)**3

def C(z):

    if z > 0:
        return (1-np.cos(z**0.5))/z

    elif z == 0:
        return 1/2.0

    else:
        return (np.cosh((-z)**0.5)-1)/-z


def find_kai(kai0,alpha,r0,vr0,Dt,mu):

    kai = kai0
    count = 0
    relax_fac = 1
    while True:
        z = alpha*kai**2

        fkai = r0*vr0/(mu)**0.5*kai**2*C(z)+(1-alpha*r0)*kai**3*S(z)+r0*kai-mu**0.5*Dt
        fkaip = r0*vr0/(mu)**0.5*kai*(1-alpha*kai**2*S(z))+(1-alpha*r0)*kai**2*C(z)+r0

        ratio = fkai/fkaip

        if np.abs(ratio)<acc33:
            break

        if count>maxiter:
            # ratio*=relax_fac
            # count = 0
            # relax_fac*=0.9
            # print("Count reset in kai")
            raise ValueError("Cannot converge in Kai search")

        if np.isnan(ratio):
            raise ValueError("ratio is nan, no way you can solve it_1")

        kai = kai-ratio
        count += 1

    return kai

def get_fgrv(alpha,r0,Dt,kai,r0_vec,v0_vec):

    f = 1-kai**2/r0*C(alpha*kai**2)
    g = Dt-1/mu**0.5*kai**3*S(alpha*kai**2)
    r_vec = f*r0_vec+g*v0_vec
    r = np.sum(r_vec**2)**0.5
    fp = mu**0.5/r/r0*(alpha*kai**3*S(alpha*kai**2)-kai)
    gp = 1 - kai**2/r*C(alpha*kai**2)
    v_vec = fp*r0_vec + gp * v0_vec

    return [f,g,r_vec,v_vec]

if __name__ == '__main__':

    # A test

    z = np.linspace(-0,500,1000)
    ss = np.zeros_like(z)
    cs = np.zeros_like(z)

    for i in range(len(z)):
        ss[i] = S(z[i])
        cs[i] = C(z[i])

