# InUCoN

<h3>Asset Subsets Identification via Investment Universe 
Complex Network</h3>

In dynamic financial markets, investors grapple with selecting the optimal subset of assets from a vast range of options. This study aims to identify an optimal subset of assets by ensuring diversity through \textit{company description} and \textit{price behavior over time}, which are two important characteristics of assets. We introduce the \textit{Investment Universe Complex Network (InUCoN)} framework, integrating these aspects to address previous limitations. The main strategy of InUCoN is to treat the investment universe as a complex system, model it with a complex network, and analyze it using this model. The framework has three components: (i) dynamic network generation; (ii) snapshot aggregation; and (iii) investment universe filtering. We selected the 200 stocks with the highest average trading volume from the SP500 starting on 01/06/2021 during two years for our experiments. We evaluate the effectiveness of InUCoN by applying the Markowitz mean-variance optimization algorithm for portfolio allocation. We compared the InUCoN with the baseline, i.e., the unfiltered universe, and with the case of using simple similarities rather than the proposed hybrid one. The results demonstrated that InUCoN effectively reduced risk by selecting a more independent set of stocks, as evidenced by improved portfolio performance metrics.

<img src="figures/commune_flow_hybrid.png"></img>
<img src="figures/allocation_results_sp.png"></img>