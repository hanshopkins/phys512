import numpy as np
import matplotlib.pyplot as plt
import initial_conditions as ic
import imageio
import os
import time

def get_rho(pos, m, dims, res): #positions, masses,size of the area, and resolution
    # dims should look like [[xmin,xmax],[ymin,ymax]]
    # res should look like [x_resolution, y_resolution]
    assert(dims[0][1] >= dims[0][0] and dims[1][1] >= dims[1][0])
    xlength = dims[0][1] - dims[0][0]
    ylength = dims[1][1] - dims[1][0]
    
    dx = xlength/res[0]
    dy = ylength/res[1]
    
    rho = np.zeros(res)
    for i in range(len(pos)):
        xidx = int(pos[i][0]/dx)
        yidx = int(pos[i][1]/dy)
        rho[yidx][xidx] += m[i]
    
    return rho

def get_kernel(n,r0, side_length): #this is stolen/modified for 2d from the example. I really can't think of a better way to write this.
    x = np.fft.fftfreq(n)*side_length
    rsqr = np.outer(np.ones(n),x**2)
    rsqr = rsqr+rsqr.T
    rsqr[rsqr<r0**2] = r0**2
    kernel = np.log(np.sqrt(rsqr))
    return kernel

def get_force_from_pot (pos, pot, dims, res):
    #this is for converting measurement units into grid units
    assert(dims[0][1] >= dims[0][0] and dims[1][1] >= dims[1][0])
    xlength = dims[0][1] - dims[0][0]
    ylength = dims[1][1] - dims[1][0]
    
    dx = xlength/res[0]
    dy = ylength/res[1]
    
    #calculating forces
    forces = np.empty([len(pos),2])
    for i in range(len(pos)):
        xidx = int(pos[i][0]/dx)
        yidx = int(pos[i][1]/dy)
        forces[i][0] = (pot[yidx][(xidx-1)%res[0]] - pot[yidx][(xidx+1)%res[0]])/(2*dx)
        forces[i][1] = (pot[(yidx-1)%res[1]][xidx] - pot[(yidx+1)%res[1]][xidx])/(2*dy)
    return forces 

##Functions for RK4
def get_derivs(xx,nnA,nnB, kernelfft,sidelength,resolution,mA,mB, G_A, G_AB, G_B):
    xA=xx[:nnA,:]%sidelength #modding side length for periodic case
    vA=xx[nnA:2*nnA,:]
    xB = xx[2*nnA:2*nnA+nnB,:]%sidelength
    vB=xx[-nnB:,:]
    
    #getting forces
    rhoA = get_rho(xA, mA, [[0,sidelength],[0,sidelength]], [resolution,resolution])
    potA = np.fft.irfft2(np.fft.rfft2(rhoA)*kernelfft, [resolution,resolution])
    rhoB = get_rho(xB, mB, [[0,sidelength],[0,sidelength]], [resolution,resolution])
    potB = np.fft.irfft2(np.fft.rfft2(rhoB)*kernelfft, [resolution,resolution])
    
    # the forces differ by constants of G
    fA = get_force_from_pot (xA, G_A * potA + G_AB * potB, [[0,sidelength],[0,sidelength]], [resolution,resolution])
    fB = get_force_from_pot (xB, G_AB * potA + G_B * potB, [[0,sidelength],[0,sidelength]], [resolution,resolution])
    
    return np.vstack([vA,fA,vB,fB])


def take_step_rk4_periodic(xA,vA,xB,vB,dt,kernelfft,sidelength,resolution,mA,mB, G_A, G_AB, G_B): #modified rk4
    nnA=xA.shape[0]
    nnB=xB.shape[0]
    xx=np.vstack([xA,vA,xB,vB]) #we need to deal with the particles separately
    k1=get_derivs(xx,nnA,nnB, kernelfft, sidelength,resolution,mA,mB, G_A, G_AB, G_B)
    k2=get_derivs(xx+k1*dt/2,nnA,nnB, kernelfft, sidelength,resolution,mA,mB, G_A, G_AB, G_B)
    k3=get_derivs(xx+k2*dt/2,nnA,nnB, kernelfft, sidelength,resolution,mA,mB, G_A, G_AB, G_B)
    k4=get_derivs(xx+k3*dt,nnA,nnB, kernelfft, sidelength,resolution,mA,mB, G_A, G_AB, G_B)
    
    tot=(k1+2*k2+2*k3+k4)/6
    
    xA = (xA+tot[:nnA,:]*dt)%sidelength #modding side length because periodic
    xB = (xB+tot[2*nnA:2*nnA+nnB,:]*dt)%sidelength
    vA=vA+tot[nnA:2*nnA,:]*dt
    vB=vB+tot[-nnB:,:]*dt
    return xA,vA,xB,vB
##################################### end of functions for RK4


def nbodysim(particlesA, particlesB, sidelength, resolution, niter, plotFramesSkip, G_A, G_AB, G_B, R0, title, outputFileName, periodic, steptype = "leapfrog", plotForces_ = False):
    #checking whether it's periodic and setting the resolution appropriately
    if not periodic:
        resolution = resolution * 2
        sidelength = sidelength * 2
        #shifting particle positions
        particlesA.positions += sidelength/4
        particlesB.positions += sidelength/4
    
    dt = 0.05
    the_kernel = get_kernel(resolution, R0, 1.0)
    kernelfft = np.fft.rfft2(the_kernel)
    
    #stuff for saving the GIF
    filenames = []
    
    t1 = time.time()
    for i in range(niter):
        if steptype == "leapfrog":
            #calculating new positions
            if periodic:
                particlesA.positions = (particlesA.positions + particlesA.velocities * dt)%sidelength #the modulus is for periodic boundary conditions
                particlesB.positions = (particlesB.positions + particlesB.velocities * dt)%sidelength
            else:
                particlesA.positions = (particlesA.positions + particlesA.velocities * dt)
                particlesB.positions = (particlesB.positions + particlesB.velocities * dt)
                #finding which particles are not in the region
                if len(particlesA.positions) > 0:
                    indexes = np.asarray((np.min(particlesA.positions, axis = 1) >= sidelength/4) * (np.max(particlesA.positions, axis = 1) < 0.75*sidelength), dtype = "bool") #there's like 25% chance this works properly
                    particlesA.positions = particlesA.positions[indexes]
                    particlesA.velocities = particlesA.velocities[indexes]
                    particlesA.masses = particlesA.masses[indexes]
                #repeating for particles B
                if len(particlesB.positions) > 0:
                    #print((np.min(particlesA.positions, axis = 1) >= sidelength/4), (np.max(particlesA.positions, axis = 1) < 0.75*sidelength))
                    indexes = np.asarray((np.min(particlesB.positions, axis = 1) >= sidelength/4) * (np.max(particlesB.positions, axis = 1) < 0.75*sidelength), dtype = "bool")
                    particlesB.positions = particlesB.positions[indexes]
                    particlesB.velocities = particlesB.velocities[indexes]
                    particlesB.masses = particlesB.masses[indexes]
                
            #calculating forces
            rhoA = get_rho(particlesA.positions, particlesA.masses, [[0,sidelength],[0,sidelength]], [resolution,resolution])
            potA = np.fft.irfft2(np.fft.rfft2(rhoA)*kernelfft, [resolution,resolution])
            rhoB = get_rho(particlesB.positions, particlesB.masses, [[0,sidelength],[0,sidelength]], [resolution,resolution])
            potB = np.fft.irfft2(np.fft.rfft2(rhoB)*kernelfft, [resolution,resolution])
            
            # the forces differ by constants of G
            forcesA = get_force_from_pot (particlesA.positions, G_A * potA + G_AB * potB, [[0,sidelength],[0,sidelength]], [resolution,resolution])
            forcesB = get_force_from_pot (particlesB.positions, G_AB * potA + G_B * potB, [[0,sidelength],[0,sidelength]], [resolution,resolution])
            
            #updating velocities
            if len(particlesA.velocities > 0):
                particlesA.velocities = particlesA.velocities + forcesA * (1/particlesA.masses)[:,None] * dt
            if len(particlesB.velocities > 0):
                particlesB.velocities = particlesB.velocities + forcesB * (1/particlesB.masses)[:,None] * dt
        
        elif steptype == "rk4" or steptype == "RK4":
            if periodic:
                particlesA.positions, particlesA.velocities, particlesB.positions, particlesB.velocities = take_step_rk4_periodic(particlesA.positions, particlesA.velocities, particlesB.positions, particlesB.velocities,dt, kernelfft, sidelength,resolution, particlesA.masses, particlesB.masses, G_A, G_AB, G_B)
                
                ### I need to recalculate these again for the energy
                rhoA = get_rho(particlesA.positions, particlesA.masses, [[0,sidelength],[0,sidelength]], [resolution,resolution])
                potA = np.fft.irfft2(np.fft.rfft2(rhoA)*kernelfft, [resolution,resolution])
                rhoB = get_rho(particlesB.positions, particlesB.masses, [[0,sidelength],[0,sidelength]], [resolution,resolution])
                potB = np.fft.irfft2(np.fft.rfft2(rhoB)*kernelfft, [resolution,resolution])
                #
            else:
                print("do non periodic") #note I couldn't figure out a good way to implement this one
        else: print("unknown step type")
        
        if i % plotFramesSkip == 0:
            ##calculating energy####################
            kineticEnergy = np.sum(particlesA.masses[:,None] * particlesA.velocities**2) + np.sum(particlesB.masses[:,None] * particlesB.velocities**2) #I'm ignoring the 1/2 for now, and I'll apply it later
            potentialEnergy = -np.sum(rhoA * (G_A * potA + G_AB * potB)) - np.sum(rhoB * (G_AB * potA + G_B * potB)) #potential energy will be off by a constant as long as the number of particles is constant because of the self-potential, but that's okay
            totalEnergy = 0.5*(kineticEnergy + potentialEnergy)
            ##end of calculating energy#############
            
            #plotting
            plt.style.use('dark_background')
            ax = plt.gca()
            ax.set_aspect("equal")
            if len(particlesA.positions) > 0:
                plt.scatter(particlesA.positions[:,0], particlesA.positions[:,1], s=10, color = "cyan")
            if len(particlesB.positions) > 0:
                plt.scatter(particlesB.positions[:,0], particlesB.positions[:,1], s=3, color = "white")
            if not periodic:
                plt.xlim(0.25*sidelength,0.75*sidelength)
                plt.ylim(0.25*sidelength,0.75*sidelength)
            else:
                plt.xlim(0,sidelength)
                plt.ylim(0,sidelength)
            plt.title(title)
            plt.xlabel('x')
            plt.ylabel('y')
            
            if plotForces_:
                if len(particlesA.positions > 1): plt.quiver(particlesA.positions.T[0], particlesA.positions.T[1], forcesA.T[0], forcesA.T[1], color = "red")
                if len(particlesB.positions > 1): plt.quiver(particlesB.positions.T[0], particlesB.positions.T[1], forcesB.T[0], forcesB.T[1], color = "red")
            
            #plotting energy
            plt.text(0.03,0.93, "Energy: "+str(totalEnergy), c = "lightcoral", transform=ax.transAxes)
            
            plt.savefig(str(i)+".png")
            plt.close()
            filenames.append(str(i)+".png")
        
        #reporting completion
        if (i%(20*plotFramesSkip) == 0) and (i != 0):
            print(str(int(100*i/niter+0.5)) + "% completed")
    
    # saving the gif
    with imageio.get_writer(outputFileName + '.gif', mode='I') as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
            os.remove(filename)
            
    print("Time it took:", time.time()-t1)

if __name__ == "__main__":
    particlesA, particlesB, sidelength = ic.mixOfBoth(100, 900, "phys512", 1) #this is where you choose initial conditions
    
    resolution = 256
    niter = 500
    plotFramesSkip = 1
    plotForces = False
    periodic = True
    steptype = "leapfrog" #either leapfrog or RK4
    
    G_A = 0.005 #force scaling factor. THere's different ones for the different possible interactions
    G_AB = 0.005
    G_B = -0.02
    R0 = 0.01
    
    title = "A and B Particles"
    outputFileName = "Extra_A_and_B"
    
    nbodysim(particlesA, particlesB, sidelength, resolution, niter, plotFramesSkip, G_A, G_AB, G_B, R0, title, outputFileName, periodic, steptype, plotForces_ = plotForces)
