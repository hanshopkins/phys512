import numpy as np

class particles:
    def __init__ (self, pos_, vel_, m_):
        self.positions = pos_; self.velocities = vel_; self.masses = m_
        
def string_to_seed(string):
    seed = 1
    for char in string:
        seed = seed * ord(char) % 2**64
    return seed

#rng = np.random.default_rng(string_to_seed("cool seed"))
#print(rng.integers(10, size = 3))


#defining particles
# N = 100
# mypos = np.empty([N,2])
# for i in range(N//2):
#     mypos[i] = [i*2/N, 0.33]
# for i in range(N//2,N):
#     mypos[i] = [(i*2-N)/N, 0.5]

# mymasses = np.ones(N)
# myvel = np.zeros([N,2])
# myparts = particles(mypos, myvel, mymasses)

emptyParticles = particles(np.asarray([]),np.asarray([]), np.asarray([]))

def diagonalish ():
    sidelength = 1
    myparts = particles(np.asarray([[0.1,0.1],[0.2,0.2],[0.65,0.65]]),np.asarray([[0,0],[0,0],[0,0]]), np.asarray([1,1,1]))
    return myparts, emptyParticles, sidelength

def one_with_velocity ():
    sidelength = 1
    myparts = particles(np.asarray([[0.5,0.5]]),np.asarray([[1,0]]), np.asarray([1]))
    return myparts, emptyParticles, sidelength

def one_in_middle ():
    sidelength = 1
    myparts = particles(np.asarray([[0.5,0.5]]),np.asarray([[0,0]]), np.asarray([1]))
    return myparts, emptyParticles, sidelength

def two_parts():
    sidelength = 1
    myparts = particles(np.asarray([[0.5,0.4],[0.5,0.6]]),np.asarray([[0,0],[0,0]]), np.asarray([1,1]))
    return myparts, emptyParticles, sidelength

def kindof_clumped():
    sidelength = 1
    myparts = particles(np.asarray([[0.5,0.4],[0.5,0.6],[0.4,0.4],[0.75,0.5],[0.6,0.6]]),np.asarray([[0,0],[0,0],[0,0],[-0.2,0],[0,0]]), np.asarray([1,1,1,1,1]))
    return myparts, emptyParticles, sidelength

def kindof_clumped_B_version():
    sidelength = 1
    myparts = particles(np.asarray([[0.5,0.4],[0.5,0.6],[0.4,0.4],[0.75,0.5],[0.6,0.6]]),np.asarray([[0,0],[0,0],[0,0],[-0.2,0],[0,0]]), np.asarray([1,1,1,1,1]))
    return emptyParticles, myparts, sidelength

def aLotOfBParticles(npart, seed):
    rng = np.random.default_rng(string_to_seed(seed))
    positions = rng.random([npart,2])
    velocities = np.zeros([npart,2])
    masses = np.ones(npart)
    myparts = particles(positions,velocities,masses)
    
    return myparts, emptyParticles, 1

def two_parts_B():
    sidelength = 1
    myparts = particles(np.asarray([[0.5,0.4],[0.5,0.6]]),np.asarray([[0,0],[0,0]]), np.asarray([1,1]))
    return emptyParticles, myparts, sidelength

def mixOfBoth(npartA, npartB, seed, sidelength):
    rng = np.random.default_rng(string_to_seed(seed))
    positionsA = rng.random([npartA,2])*sidelength
    velocitiesA = np.zeros([npartA,2])
    massesA = np.ones(npartA) * 11.11
    mypartsA = particles(positionsA,velocitiesA,massesA)
    
    positionsB = rng.random([npartB,2])*sidelength
    velocitiesB = np.zeros([npartB,2])
    massesB = np.ones(npartB)
    mypartsB = particles(positionsB,velocitiesB,massesB)
    return mypartsA, mypartsB, sidelength

def two_particles_circle():
    myparts = particles(np.asarray([[0.3,0.5],[0.7,0.5]]),np.asarray([[0,-np.sqrt(0.1/2)],[0,np.sqrt(0.1/2)]]), np.ones(2))
    return myparts, emptyParticles, 1