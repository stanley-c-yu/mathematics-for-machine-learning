import numpy as np 

class PCA: 
    '''
    Refactored submission for the PCA Course by the Imperial College of London on Coursera.  
    
    Many thanks to the fellow students who completed this course and provided useful tips 
    and guidance on the forums.  None of this would have been possible without you.  
    
    Please feel free to use this code as a guide, but please respect the honor code and don't blindly copy-paste. 
    '''
    
    def normalize(self, X):
        """
        Normalize the given dataset X to have zero mean.
        
        Args:
            X: ndarray, dataset of shape (N,D)
        
        Returns:
            (Xbar, mean): tuple of ndarray, Xbar is the normalized dataset
            with mean 0; mean is the sample mean of the dataset.
        """
        mu = X.mean(0)
        Xbar = X - mu
        return Xbar, mu
    
    def eig(self, S):
        """
        
        Normalize the given dataset X to have zero mean.
        
        Args:
            X: ndarray, dataset of shape (N,D)
        
        Returns:
            (Xbar, mean): tuple of ndarray, Xbar is the normalized dataset
            with mean 0; mean is the sample mean of the dataset.
        """  
        eigvals, eigvecs = np.linalg.eig(S)
        # get the indices to sort in descending order with respect to eigenvalues
        sorted_indices = np.argsort(eigvals)[::-1]
        # Note, only the columns of eigvecs is being sorted, since the columns are the eigenvectors
        return eigvals[sorted_indices], eigvecs[:, sorted_indices]
    
    def projection_matrix(self, B):
        """
        
        Compute the projection matrix onto the space spanned by `B`
        
        
        Args:
            B: ndarray of dimension (D, M), the basis for the subspace
        
        Returns:
            P: the projection matrix
        """
        P =  B @ np.linalg.inv(B.T @ B) @ B.T
        return P
        
    
    def pca(self, X, num_components):
        """
        Args:
            X: ndarray of size (N, D), where D is the dimension of the data,
               and N is the number of datapoints
            num_components: the number of principal components to use.
        Returns:
            the reconstructed data, the sample mean of the X, principal values
            and principal components
        """
        N, D = X.shape
        
        X_normalized, mean = self.normalize(X)
        
        # Compute data covariance matrix 
        S = np.cov(X_normalized/N, rowvar=False, bias=True)
        
        # Compute the eigenvalues and corresponding eigenvectors for S
        eig_vals, eig_vecs = self.eig(S)
        
        # Take the top number of components of the eigenvalues and vectors
        # AKA, the principal values and pincipal components 
        principal_vals = eig_vals[:num_components]
        principal_components = eig_vecs[:, :num_components]
        
        # Reconstruct the data using the basis spanned by the PCs.
        # Recall that the mean was subtracted from X, so it needs to be added back here
        P = self.projection_matrix(principal_components)  # projection matrix
        reconst = (P @ X_normalized.T).T + mean
        return reconst, mean, principal_vals, principal_components

    def pca_high_dim(self, X, num_components): 
        """
        Compute PCA for small sample size but high-dimensional features. 
        
        Args:
            X: ndarray of size (N, D), where D is the dimension of the sample,
               and N is the number of samples
            num_components: the number of principal components to use.
        Returns:
            X_reconstruct: (N, D) ndarray. the reconstruction
            of X from the first `num_components` pricipal components.
        """
        N, D = X.shape
        # Normalize the dataset
        X_normalized, mean = self.normalize(X)
        
        # Find the covariance matrix
        M = np.dot(X_normalized, X_normalized.T)/N
        
        # Get the eigenvalues and eigenvectors
        eig_vals, eig_vecs = self.eig(M)
        
        # Take the top number of the eigenvalues and eigenvectors
        principal_values = eig_vals[:num_components]
        principal_components = eig_vecs[:, :num_components]
        
        # reconstruct the images from the lower dimensional representation
        # Remember to add back the sample mean
        P = self.projection_matrix(principal_components)
        reconst = (P @ X_normalized) + mean
        return reconst, mean, principal_values, principal_components

        
            
        
