import numpy as np
# Note that the data have been normalized and are now in [0,1]

# Capture information about the intensity distribution along each row
def Hist_im(image, n_bins):
  ''' 
  Input : 
  - image : numpy.array of shape 32x32 pixels
  - n_bins : int 
  Output :
  - Hist : histogram normalized of shape (384,1)
  '''
    N, _ = image.shape
    Hist = []
    # For each row compute a histogram and normalize it
    for i in range(N):
        hist, _ = np.histogram(image[i , :], range=(0.2, 0.8), bins=n_bins)
        if np.sum(hist)==0:
          hist_values_scaled = hist
        else :
          hist_values_scaled = hist / np.sum(hist)
        Hist.append(hist_values_scaled)
    Hist = np.array(Hist)
    return Hist.ravel()

# Compute normalize histogram of colors of an image represented as an array of shape (3072,1)
def Colors_Hist(X_j , n_bins):
    ''' 
    Input : 
    - X_j : numpy.array of shape (3072,1) which represented pixels intensities for each colors of a 32x32 image 
    - n_bins : int
    Output :
    - Hist : histogram normalized of shape (24,1)
    ''' 
    hist_red_channel_values, _ = np.histogram(X_j[:1024] , range=(0.2, 0.8), bins=n_bins)
    hist_red_channel_values_scaled = hist_red_channel_values / np.sum(hist_red_channel_values)
  
    hist_green_channel_values, _ = np.histogram(X_j[1024:2048] , range=(0.2,0.8), bins=n_bins)
    hist_green_channel_values_scaled = hist_green_channel_values / np.sum(hist_green_channel_values)
  
    hist_blue_channel_values, _ = np.histogram(X_j[2048:] , range=(0.2, 0.8), bins=n_bins)
    hist_blue_channel_values_scaled = hist_blue_channel_values / np.sum(hist_blue_channel_values)
  
    Hist = np.concatenate([hist_red_channel_values_scaled, hist_green_channel_values_scaled, hist_blue_channel_values_scaled])
    return Hist
