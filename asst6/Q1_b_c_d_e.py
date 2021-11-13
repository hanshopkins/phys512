import numpy as np
import matplotlib.pyplot as plt
from Q1_a import create_noise_model
from Q1_a import Tukey_Window
from read_ligo import read_file
from read_ligo import read_template

directory = "LOSC_Event_tutorial/LOSC_Event_tutorial/"

def matched_filter (data, template, noise, window_array):
    data = data * window_array
    template = template * window_array
    
    dft = np.fft.rfft(data)
    tft = np.fft.rfft(template)
    
    return(np.fft.irfft(dft*np.conj(tft)/noise))

if __name__ == "__main__":
    ##first event
    strain_H1 = read_file(directory + "H-H1_LOSC_4_V1-1167559920-32.hdf5")[0]
    strain_L1 = read_file(directory + "L-L1_LOSC_4_V1-1167559920-32.hdf5")[0]
    
    th1,tl1 = read_template(directory + "GW170104_4_template.hdf5")
    
    noise_model_H1 = create_noise_model(strain_H1)
    noise_model_L1 = create_noise_model(strain_L1)
    
    N = len(strain_H1)
    window_values = np.empty(N)
    for n in np.arange(N):
        window_values[n] = Tukey_Window(n,N,0.5)
        
    result_H1 = matched_filter (strain_H1, th1, noise_model_H1, window_values)
    result_L1 = matched_filter (strain_L1, tl1, noise_model_L1, window_values)
    
    fig1, axs1 = plt.subplots(2)
    fig1.suptitle("Searching for a detection Event V1-1167559920")
    axs1[0].plot(np.fft.fftshift(result_H1))
    axs1[0].set_title("Hanford")
    axs1[0].get_xaxis().set_visible(False)
    axs1[1].plot(np.fft.fftshift(result_L1))
    axs1[1].set_title("Livingston")
    axs1[1].get_xaxis().set_visible(False)
    plt.savefig(r"event1_detection.pdf")
    
    ##second event
    strain_H2 = read_file(directory + "H-H1_LOSC_4_V2-1126259446-32.hdf5")[0]
    strain_L2 = read_file(directory + "L-L1_LOSC_4_V2-1126259446-32.hdf5")[0]
    
    th2,tl2 = read_template(directory + "GW150914_4_template.hdf5")
    
    noise_model_H2 = create_noise_model(strain_H2)
    noise_model_L2 = create_noise_model(strain_L2)
    
    result_H2 = matched_filter (strain_H2, th2, noise_model_H2, window_values)
    result_L2 = matched_filter (strain_L2, tl2, noise_model_L2, window_values)
    
    fig2, axs2 = plt.subplots(2)
    fig2.suptitle("Searching for a detection Event V2-1126259446")
    axs2[0].plot(np.fft.fftshift(result_H2))
    axs2[0].set_title("Hanford")
    axs2[0].get_xaxis().set_visible(False)
    axs2[1].plot(np.fft.fftshift(result_L2))
    axs2[1].set_title("Livingston")
    axs2[1].get_xaxis().set_visible(False)
    plt.savefig(r"event2_detection.pdf")
    
    ##third event
    strain_H3 = read_file(directory + "H-H1_LOSC_4_V2-1128678884-32.hdf5")[0]
    strain_L3 = read_file(directory + "L-L1_LOSC_4_V2-1128678884-32.hdf5")[0]
    
    th3,tl3 = read_template(directory + "LVT151012_4_template.hdf5")
    
    noise_model_H3 = create_noise_model(strain_H3)
    noise_model_L3 = create_noise_model(strain_L3)
    
    result_H3 = matched_filter (strain_H3, th3, noise_model_H3, window_values)
    result_L3 = matched_filter (strain_L3, tl3, noise_model_L3, window_values)
    
    fig3, axs3 = plt.subplots(2)
    fig3.suptitle("Searching for a detection Event V2-1128678884")
    axs3[0].plot(np.fft.fftshift(result_H3))
    axs3[0].set_title("Hanford")
    axs3[0].get_xaxis().set_visible(False)
    axs3[1].plot(np.fft.fftshift(result_L3))
    axs3[1].set_title("Livingston")
    axs3[1].get_xaxis().set_visible(False)
    plt.savefig(r"event3_detection.pdf")
    
    ##fourth event
    strain_H4 = read_file(directory + "H-H1_LOSC_4_V2-1135136334-32.hdf5")[0]
    strain_L4 = read_file(directory + "L-L1_LOSC_4_V2-1135136334-32.hdf5")[0]
    
    th4,tl4 = read_template(directory + "GW151226_4_template.hdf5")
    
    noise_model_H4 = create_noise_model(strain_H4)
    noise_model_L4 = create_noise_model(strain_L4)
    
    result_H4 = matched_filter (strain_H4, th4, noise_model_H4, window_values)
    result_L4 = matched_filter (strain_L4, tl4, noise_model_L4, window_values)
    
    fig4, axs4 = plt.subplots(2)
    fig4.suptitle("Searching for a detection Event V2-1135136334")
    axs4[0].plot(np.fft.fftshift(result_H4))
    axs4[0].set_title("Hanford")
    axs4[0].get_xaxis().set_visible(False)
    axs4[1].plot(np.fft.fftshift(result_L4))
    axs4[1].set_title("Livingston")
    axs4[1].get_xaxis().set_visible(False)
    plt.savefig(r"event4_detection.pdf")
    
    ########################
    # this is part c
    #I want to cut off either side of the shape since they go to 0 and it doesn't accurately reflect the noise. The window is 1 in the entire middle half, so I'll just consider that.
    n1 = len(result_H1)//4; n2 = len(result_H1)*3//4
    noise_H1 = np.std(result_H1[n1:n2]) #estimating the noise is just the std
    signal_height_H1 = np.max(np.abs(result_H1)) #this isn't perfect since there could be another spike higher than the spike we want, but they'll be about the same height anyway so it's fine.
    print("The signal to noise ratio of H1 is", signal_height_H1/noise_H1)
    
    noise_L1 = np.std(result_L1[n1:n2])
    signal_height_L1 = np.max(np.abs(result_L1))
    print("The signal to noise ratio of L1 is", signal_height_L1/noise_L1)
    
    #For the combined, I think that just means adding them?
    print("The combined signal to noise ratio for event 1 is", np.sqrt((signal_height_H1/noise_H1)**2 + (signal_height_L1/noise_L1)**2))
    
    noise_H2 = np.std(result_H2[n1:n2])
    signal_height_H2 = np.max(np.abs(result_H2))
    print("The signal to noise ratio of H2 is", signal_height_H2/noise_H2)
    
    noise_L2 = np.std(result_L2[n1:n2])
    signal_height_L2 = np.max(np.abs(result_L2))
    print("The signal to noise ratio of L2 is", signal_height_L2/noise_L2)
    
    print("The combined signal to noise ratio for event 2 is", np.sqrt((signal_height_H2/noise_H2)**2 + (signal_height_L2/noise_L2)**2))
    
    noise_H3 = np.std(result_H3[n1:n2])
    signal_height_H3 = np.max(np.abs(result_H3))
    print("The signal to noise ratio of H3 is", signal_height_H3/noise_H3)
    
    noise_L3 = np.std(result_L3[n1:n2])
    signal_height_L3 = np.max(np.abs(result_L3))
    print("The signal to noise ratio of L3 is", signal_height_L3/noise_L3)
    
    print("The combined signal to noise ratio for event 3 is", np.sqrt((signal_height_H3/noise_H3)**2 + (signal_height_L3/noise_L3)**2))
    
    noise_H4 = np.std(result_H4[n1:n2])
    signal_height_H4 = np.max(np.abs(result_H4))
    print("The signal to noise ratio of H4 is", signal_height_H4/noise_H4)
    
    noise_L4 = np.std(result_L4[n1:n2])
    signal_height_L4 = np.max(np.abs(result_L4))
    print("The signal to noise ratio of L4 is", signal_height_L4/noise_L4)
    
    print("The combined signal to noise ratio for event 4 is", np.sqrt((signal_height_H4/noise_H4)**2 + (signal_height_L4/noise_L4)**2))
    
    
    print("\n")
    ## And here's part D:
    th1_fft = np.fft.rfft(th1*window_values)
    analytic_snrsq_h1 = np.real(np.sum(th1_fft*np.conj(th1_fft)/noise_model_H1))/N
    print("H1 Analytic SNR:", np.sqrt(analytic_snrsq_h1))
    tl1_fft = np.fft.rfft(tl1*window_values)
    analytic_snrsq_l1 = np.real(np.sum(tl1_fft*np.conj(tl1_fft)/noise_model_L1))/N
    print("L1 Analytic SNR:", np.sqrt(analytic_snrsq_l1))
    th2_fft = np.fft.rfft(th2*window_values)
    analytic_snrsq_H2 = np.real(np.sum(th2_fft*np.conj(th2_fft)/noise_model_H2))/N
    print("H2 Analytic SNR:", np.sqrt(analytic_snrsq_H2))
    tl2_fft = np.fft.rfft(tl2*window_values)
    analytic_snrsq_L2 = np.real(np.sum(tl2_fft*np.conj(tl2_fft)/noise_model_L2))/N
    print("L2 Analytic SNR:", np.sqrt(analytic_snrsq_L2))
    th3_fft = np.fft.rfft(th3*window_values)
    analytic_snrsq_H3 = np.real(np.sum(th3_fft*np.conj(th3_fft)/noise_model_H3))/N
    print("H3 Analytic SNR:", np.sqrt(analytic_snrsq_H3))
    tl3_fft = np.fft.rfft(tl3*window_values)
    analytic_snrsq_L3 = np.real(np.sum(tl3_fft*np.conj(tl3_fft)/noise_model_L3))/N
    print("L3 Analytic SNR:", np.sqrt(analytic_snrsq_L3))
    th4_fft = np.fft.rfft(th4*window_values)
    analytic_snrsq_H4 = np.real(np.sum(th4_fft*np.conj(th4_fft)/noise_model_H4))/N
    print("H4 Analytic SNR:", np.sqrt(analytic_snrsq_H4))
    tl4_fft = np.fft.rfft(tl4*window_values)
    analytic_snrsq_L4 = np.real(np.sum(tl4_fft*np.conj(tl4_fft)/noise_model_L4))/N
    print("L4 Analytic SNR:", np.sqrt(analytic_snrsq_L4))
    
    print("\n")
    ##here's part e
    def part_e_result_index(template_fft, noise_model, analytic_snrsq):
        cumsum = np.cumsum(np.real(template_fft*np.conj(template_fft)/noise_model/N))
        for i in range(len(cumsum)):
            if cumsum[i] > analytic_snrsq/2:
                return i
    print("Part e H1:", part_e_result_index(th1_fft, noise_model_H1, analytic_snrsq_h1))
    print("Part e L1:", part_e_result_index(tl1_fft, noise_model_L1, analytic_snrsq_l1))
    print("Part e H2:", part_e_result_index(th2_fft, noise_model_H2, analytic_snrsq_H2))
    print("Part e L2:", part_e_result_index(tl2_fft, noise_model_L2, analytic_snrsq_L2))
    print("Part e H3:", part_e_result_index(th3_fft, noise_model_H3, analytic_snrsq_H3))
    print("Part e L3:", part_e_result_index(tl3_fft, noise_model_L3, analytic_snrsq_L3))
    print("Part e H4:", part_e_result_index(th4_fft, noise_model_H4, analytic_snrsq_H4))
    print("Part e L4:", part_e_result_index(tl4_fft, noise_model_L4, analytic_snrsq_L4))