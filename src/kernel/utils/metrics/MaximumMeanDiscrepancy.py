def MaximumMeanDiscrepancy(X,Y,kernel):
    '''
    Compute Maximum Mean Discrepancy:
    ||m_X-m_Y||^2 = 1/(N^2) k(X,X) + 1/(M^2) k(Y,Y) - 2*1/(NM) k(X,Y)
    '''
    MMD = kernel.pdf(X,X).mean()+kernel.pdf(Y,Y).mean()-2*kernel.pdf(X,Y).mean()
    return MMD
