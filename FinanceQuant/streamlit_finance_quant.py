# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 10:56:36 2023

@author: giuli
"""

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from collections import OrderedDict
from PIL import Image
from pypfopt import plotting
from pypfopt.efficient_frontier import EfficientFrontier
from scipy.stats import kurtosis, skew
import warnings
warnings.filterwarnings("ignore")
from scipy.cluster.hierarchy import linkage,dendrogram,fcluster
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

def plot_line(ticker, df):
    ticker_data = df[df['variable'] == ticker]
    fig, ax = plt.subplots()
    ax.plot(ticker_data['Date'], ticker_data['value'])
    ax.axhline(y=0, color='black', linestyle='-')
    ax.set_title("Rendement mensuelle de l'action {} (2017-2022)".format(ticker))
    ax.set_xlabel("Date")
    ax.set_ylabel("Rendements (%)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    return fig

def plot_histogram(ticker, df):
    ticker_data = df[df['variable'] == ticker]
    plt.hist(ticker_data['value'], color='red')
    plt.title(f"Histogramme des rendements pour {ticker}")
    plt.xlabel("Rendements")
    plt.ylabel("Fréquence")
    return plt

#Importation des données 
df_return=pd.read_excel(r"./FinanceQuant/Data_avec_feuille_T.xlsx", sheet_name="Rendements")
df_return_correlation=pd.read_excel(r"./FinanceQuant/Data projet.xlsx", sheet_name="Returns")
df_return_correlation=df_return_correlation.set_index("Stock").T
df_return_correlation.index = pd.to_datetime(df_return_correlation.index)
# Traitement des valeurs manquantes 
df_return_correlation.drop(columns=["DOW US Equity","CTVA US Equity"],axis=1,inplace=True)
df_return_correlation = df_return_correlation.bfill(axis='rows')
df_return_correlation = df_return_correlation.ffill(axis='rows')
# Traitement des valeurs manquantes 
df_return = df_return.bfill(axis='columns')
df_return = df_return.ffill(axis='columns')

#pour les graphiques partie 1 
df=df_return.melt(id_vars="Date")


#Creation d'un bouton radio pour selectionner les menus qui correspondent à chaque partie du projet
menu = st.sidebar.radio("Navigation", ["Description des données", "Clustering", "Moyenne-Variance", "Portfolio construction and monitoring"])

#PREMIERE PARTIE: description des données
if menu == "Description des données":
    #Creation d'un sous menu dans la partie description des données
    sub_menu = st.sidebar.radio("Sous-section Description des données", ["Statistiques", "Performance cumulée des titres (base100)", "Matrice des corrélations", "Indicateurs synthétiques du risque"])
    if sub_menu == "Statistiques":
        st.header("Statistiques descriptives : ")
        #Creation d'une multiselection pour selectionner les actions à afficher
        selected_stocks = st.multiselect("Selectionner la/les action(s) : ", df['variable'].unique())
        
        #Creation d'une select box pour selectionner le type de graphique souhaite
        plot_type = st.selectbox("Selectionner le type de graphique souhaité : ", ["Ligne", "Histogramme"])
    
        #Boucle qui parcours les actions selectionnees et il affiche les statistiques et le graphique souhaite
        for ticker in selected_stocks:
            st.header(f"{ticker} statistiques et graphique")
            df['value'] = df['value'].astype(float)
            #Creation d'un dataframe qui contient toutes les statistiques 
            grouped_data = df[df['variable'] == ticker].groupby('variable')
            summary_stats = grouped_data.agg({'value': ['min', 'max', 'mean', 'var', 'std']}).round(4)
            summary_stats['Kurtosis'] = grouped_data['value'].apply(lambda x: kurtosis(x))
            summary_stats['Skewness'] = grouped_data['value'].apply(lambda x: skew(x))
            summary_stats.columns = ['Minimum', 'Maximum', 'Moyenne', 'Variance', 'Volatilite', 'Kurtosis', 'Skewness']
            #Affichage des statistiques
            st.table(summary_stats)
    
            #Affichage du graphique choisi pour chaque action
            if plot_type == "Ligne":
                fig = plot_line(ticker, df)
                st.pyplot(fig)
            else:
                fig = plot_histogram(ticker, df)
                st.pyplot(fig)
                
    elif sub_menu == "Performance cumulée des titres (base100)":
        st.header("Performance cumulée des titres (base100)")
        #importer la données base 100
        df_return_base100=pd.read_excel(r"./FinanceQuant/Data_avec_feuille_T.xlsx", sheet_name="cumule")
        df_base100=df_return_base100.melt(id_vars="Date")
        fig = plt.figure(figsize=(22, 16))
        #Creation du graphique base 100
        for variable in df_base100['variable'].unique():
            subset = df_base100[df_base100['variable'] == variable]
            plt.plot(subset['Date'], subset['value'],label = variable)

        # Ajout du titre et des labels pour l'axe des x et des y
        plt.title("Rendement cumulé des actifs pour la période 2017 - 2021 (base 100 01/12/2017)")
        plt.xlabel("Date")
        plt.ylabel("Rendement")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0., ncol=3)
        #Affichage du graphique
        st.pyplot(plt)
        
        st.write("<div style='text-align: justify'>L'interprétation d'un graphique de performance cumulée des actions en base 100 permet d’avoir un aperçu de la croissance ou de la baisse de la valeur d'une action particulière sur une période donnée. Un graphique de base 100 représente la valeur initiale du stock sous la forme 100 et affiche la variation de la valeur au fil du temps sous la forme d'un pourcentage d'augmentation ou de diminution par rapport à la valeur initiale.</div>", unsafe_allow_html=True)
        st.write("<div style='text-align: justify'>Si la courbe augmente, cela indique que la valeur de l’action a augmenté, inversement elle a diminué. </div>", unsafe_allow_html=True)
        st.write("<div style='text-align: justify'>A travers le graphique nous pouvons donc constater que à partir de la fin de 2017, les actions étudiées ont connu une baisse jusqu’à le premier semestre de 2020 et cela s’explique par le phénomène inattendu de la crise Covid19. A partir de ce moment-là, nous pouvons voir que l’ensemble des actions n’a pas cessé d’augmenter mais certaines sont revenus à leur niveau de fin 2017.</div>", unsafe_allow_html=True)
        pass
    
    elif sub_menu=="Matrice des corrélations":
        # Calcul de la correlation entre TOUTES les actions
        df_correlations = df_return_correlation.corr()
        #Affichage de la matrice des corrélations 
        st.header("Matrice de corrélation pour toutes les actions")
        st.write(df_correlations.style.background_gradient(), unsafe_allow_html=True)
        
        st.markdown("<div style='text-align: justify'>A travers la matrice de corrélation ci-dessus , on peut tout d'abord constater que les fonds appartenant aux mêmes secteurs (energy, basic material, consumer non cyclial) et sont corrélés à plus de 50% même si ils ne sont pas situés dans le même pays. Par exemple, les actions <b>BP/LN Equity</b> et <b>XOM US Equity</b> sont corrélés à 74% car elles appartiennent toutes deux au secteur de l'énergie et sont de plus toutes deux dans le secteur pétrolier. Egalement, on peut voir que des fonds dans le secteur de l'énergie sont corrélés à des fonds du secteur <b>Basic Material</b> qui  évoluent dans les mêmes domaines d'activités. C'est le cas pour les actions <b>BP/LN Equity</b> et <b>AAL LN Equity</b> qui évoluent respectivement dans le raffinage de pétrole et la transformation minière. On remarque également que les actions appartenant au secteur <b>Consumer non cyclical</b> sont corrélés à des actions portant sur la même devise indépendamment de leurs secteurs respectifs.  Enfin, on remarque que les actions du secteur <b>Basic materialé</b> sont globalement plus ou moins corrélés avec les actions des autres secteurs même si pour celles il y a une légère hausse de corrélations lorsque les actions portent sur la même devise. Ces constats s'expliquent très bien par le fait que les actions portant sur les mêmes domaines d'activités sont affectés par les mêmes évènements (baisse du cours du pétrole) tout comme celles portant sur les mêmes devises.</div>", unsafe_allow_html=True)
        
        # Creation d'une multi selection pour selectionner les actions pour lesquelles ont veut voir la correlation
        selected_stocks = st.multiselect("Selectionner la/les action(s) : ", df_return_correlation.columns)

        #Filtre sur le dataframe pour inclure que les actions selectionnees
        df_returns_filtered = df_return_correlation[selected_stocks]

        #Calcul des correlations sur le dataframe filtre
        correlation_matrix = df_returns_filtered.corr()

       #Affichage de la matrice des corrélations filtrees 
        st.write("Matrice des corrélations pour les actions sélectionnées :", correlation_matrix)
    
    else:
        st.header("Indicateurs synthétiques du risque : Ratio de Sharpe et VaR ")
        st.markdown("<div style='text-align: justify'><p><b>Ratio de Sharpe</b></p></div>", unsafe_allow_html=True)
        st.markdown("<div style='text-align: justify'><p>Le Ratio de Sharpe mesure la rentabilité d'un portefeuille par rapport au risque pris par l'investisseur. </p></div>", unsafe_allow_html=True)
        st.markdown("<div style='text-align: justify'><p>- Si le ratio de Sharpe < 0, cela signifie que les actions formant notre portefeuille d'actifs ne sont pas performantes.</p></div>", unsafe_allow_html=True)
        st.markdown("<div style='text-align: justify'><p>- Si le ratio est compris entre 0 et 1, cela signifie que la rentabiité espérée du portefeuille d'actifs est plus importante qu'un actif sans risque mais que les risques engagés sont plus grands que l'excédent de rendements.</p></div>", unsafe_allow_html=True)
        st.markdown("<div style='text-align: justify'><p>- Si le ratio de Sharpe > 1, cela signifie que le portefeuille est performant et que l'excédent de rendements est supérieur au risque pris.</p></div>", unsafe_allow_html=True)
        st.markdown("<div style='text-align: justify'><p>On remarque que le ratio de Sharpe de l'ensemble des actions est compris entre 0 et 1 ce qui signifie que la rentabilité espérée des actions est plus important que celle d'un actif sans risque.</p></div>", unsafe_allow_html=True)
        st.markdown("<div style='text-align: justify'><p><b>VaR</b></p></div>", unsafe_allow_html=True)
        st.markdown("<div style='text-align: justify'><p>La Value at Risk historique permet de connaître la perte potentielle maximale que l'on peut atteindre pour un intervalle de confiance et une période donnée.</p></div>", unsafe_allow_html=True)
        
        selected_stocks = st.multiselect("Selectionner la/les action(s) : ", df_return.columns)
        
        if selected_stocks:
            for stock in selected_stocks:
                stock_return = df_return[stock].mean(axis=0)
                stock_std = df_return[stock].std(axis=0)
                stock_sharpe_ratio = stock_return / stock_std
                stock_var = np.percentile(df_return[stock], 5, axis=0, interpolation="lower")
        
                st.markdown(f"**Ratio de Sharpe** pour {stock}: {stock_sharpe_ratio}")
                st.markdown(f"**VaR** pour {stock} : {stock_var}")
                st.markdown("---")



elif menu == "Clustering":
    sub_menu_clustering = st.sidebar.radio("Sous-section Clustering", ["K-Means returns", "CAH ROI"])
    if sub_menu_clustering == "K-Means returns":
        st.header("Clustering à l'aide de l'algorithme K-Means")
        df_return=pd.read_excel(r"./FinanceQuant/Data_projet_finance.xlsx", sheet_name="Returns")
        
        st.write("<div style='text-align: justify'><p>Dans cette partie, nous allons donc implémenter l'algorithme de clustering K-means sur les rendements des 58 actions dans la base de données. Cette partie nous permettra d'afficher les différents résultats obtenus : les méthodologies utilisées seront évoqués plus en détail dans le rapport rédigé</div>", unsafe_allow_html=True)
        
        df = df.set_index(df.columns[0])
        df = df.drop(df.columns[0], axis=1)
        df = df.bfill(axis='columns')
        df = df.ffill(axis='columns')
        test = df.columns
        
        #On choisit un premier paramètre : K=3
        K = 3
        from sklearn.cluster import KMeans
        clf = KMeans(n_clusters=K, n_init=10, random_state=0)
        
        Xbis = df[test].values #récupérations de données
        from sklearn.preprocessing import StandardScaler 
        sc = StandardScaler()
        X_normalisebis = sc.fit_transform(Xbis) #on normalise alors les données
        clf.fit(X_normalisebis)
        #on applique donc la méthode des K-means utilisé sur notre base de données
        preds = clf.predict(X_normalisebis)
        
        st.write("<div style='text-align: justify'><p>Après avoir imputé les valeurs manquantes et normalisé les données, nous chercherons donc à implémenter l'algorithme K-means sur ces données. Pour réaliser cela, il est tout d'abord nécessaire de déterminer le nombre de clusters optimal à définir.</div>", unsafe_allow_html=True)
        
        st.write("<div style='text-align: justify'><p>Pour faire cela, on va commencer par utiliser la méthode du coude (elbow method). Cette méthode consiste à calculer le score associé à chaque valeur potentiel de K et de représenter cela sur un graphique. Une fois cela réalisé, le nombre de clusters optimal correspondra à l'instant où la courbe forme un angle droit.</div>", unsafe_allow_html=True)
        
        st.write("<div style='text-align: justify'><p>Le graphique est représenté ci-dessous : </div>", unsafe_allow_html=True)

#Puis, on essaye de trouver le nombre de clusters optimal
#1/ Etablissement de la méthode du coude 
        score = []

        for i in range(2,9+1):
            km = KMeans(n_clusters=i,random_state=0, n_init=10).fit(X_normalisebis)
            preds = km.predict(X_normalisebis)
            score.append(-km.score(X_normalisebis))
    
        #détermination du score pour chacune des valeurs de K entre 2 et 10
        #Creation d'un graphique regroupant l'ensemble des résultats de la cellule précédente
        import numpy as np 
        import pandas as pd
        import seaborn as sns
        import random
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12,5))
        plt.title("Critère du coude",fontsize=16)
        plt.plot(range(2,10),score,marker='o')
        plt.grid(True)
        plt.xlabel('Nombre de clusters',fontsize=14)
        plt.ylabel('K-means score',fontsize=14)
        plt.xticks(range(2,10),fontsize=14)
        plt.yticks(fontsize=15)
        st.pyplot(plt)
        st.set_option('deprecation.showPyplotGlobalUse', False)

        st.write("<div style='text-align: justify'><p>Nous remarquons alors qu'aucun angle droit ne se distingue clairement sur ce graphique. Par conséquent, nous allons privilégier une autre méthode pour déterminer le nombre de clusters optimal pour cette base de données.</div>", unsafe_allow_html=True)
        
        st.write("<div style='text-align: justify'><p>La méthode choisie sera celle utilisant le coefficient Silhouette. L'idée est de calculer le coefficient Silhouette pour chacune des valeurs potentielles de K. Une fois cela réalisé, le nombre optimal de clusters sera la valeur de K pour laquelle cette métrique est maximisé. Les calculs du coefficient Silhouette pour chacun des nombres de clusters possibles sont représentés ci-dessous : </div>", unsafe_allow_html=True)
        
        from PIL import Image
        # image = Image.open("test11113.png")

        st.write("<div style='text-align: justify'><p>On remarque que le coefficient Silhouette est maximisé pour K=3. Par conséquent, le nombre de clusters optimal pour cette base de données selon l'algorithme K-means est de 3.</div>", unsafe_allow_html=True)

        st.write("<div style='text-align: justify'><p>Une fois l'algorithme K-means implémenté, il est possible de représenter les 3 différents clusters sur un graphique. Nous avons choisi de les représenter sur un plan mettant en relation les deux premières composantes d'une analyse en composante principales. Les différents clusters sont caractérisés par des couleurs différentes.</div>", unsafe_allow_html=True)
        
        df = pd.read_excel(r"./FinanceQuant/Data_projet_finance.xlsx", sheet_name='Returns')
        
        df = df.set_index(df.columns[0]) #remplacement du nom des colonnes
    
        df = df.drop(df.columns[0], axis=1) #suppression de la première colonne
        df = df.bfill(axis='columns')
        df = df.ffill(axis='columns')
        
        test = df.columns #récupération d'une liste contenant le nom des colonnes
        Xbis = df[test].values #récupérations de données

        from sklearn.preprocessing import StandardScaler 
        sc = StandardScaler()
        
        X_normalise = sc.fit_transform(Xbis) #on normalise alors les données
        
        from sklearn.decomposition import PCA
        acp = PCA()
        
        acp.fit(X_normalise) #application de l'ACP

        
        new_columns = acp.transform(X_normalise)

        df_new = pd.DataFrame(new_columns,columns=[f'pca_{i}' for i in range(1,60)]) 
        
        plt.figure(figsize=(18, 6))
        plt.scatter(new_columns[:, 0], new_columns[:, 1])

        for label, x, y in zip(df.index.values, new_columns[:, 0], new_columns[:, 1]):
            plt.annotate(
                label,
                xy=(x, y), xytext=(-5, 5),
                textcoords='offset points', ha='right', va='bottom',
                bbox=dict(boxstyle='round,pad=0.5', fc='red', alpha=0.5),
                arrowprops = dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    

        plt.title("Représentation de l'ensemble des actions sur le plan de l'ACP");
        st.pyplot(plt)
        
        st.write("<div style='text-align: justify'><p>Cependant, il semble plus intéressant de pouvoir les distinguer selon leurs clusters d'appartenance pour s'assurer de la cohérence du clustering, ce qu'on représente ci-dessous</div>", unsafe_allow_html=True)
        
        n,p = X_normalise.shape #détermination du nombre d'observations et de colonnes
        eigval = ((n-1) / n) * acp.explained_variance_ # détermination des valeurs propres
        sqrt_eigval = np.sqrt(eigval) # détermination de la racine carrée des valeurs propres
        corvar = np.zeros((p,p)) # définition de la matrice vide
        for k in range(p):
            corvar[:,k] = acp.components_[k,:] * sqrt_eigval[k]

        #formation du dataframe regroupant l'ensemble de ces informations
        df_coord = pd.DataFrame({'id': test, 'corr_axe1': corvar[:,0], 'corr_axe2': corvar[:,1]})

        correlations_axe1 = []
        correlations_axe2 = []

        for i,feat in enumerate(test):
            corr_axe1 = np.corrcoef(X_normalise[:,i],df_new['pca_1'].values)[0][1]
            corr_axe2 = np.corrcoef(X_normalise[:,i],df_new['pca_2'].values)[0][1]
            correlations_axe1.append(corr_axe1)
            correlations_axe2.append(corr_axe2)
    
        df_coord = pd.DataFrame({'id': test, 'corr_axe1': correlations_axe1, 'corr_axe2': correlations_axe2})

        #On réalise donc le clustering K-means avec 3 clusters 
        km = KMeans(n_clusters=3, n_init=10,random_state=0).fit(X_normalise)
        preds = km.predict(X_normalise)

        #On assigne également à chaque commune son numéro de cluster
        df['cluster'] = preds
        
        #Il est donc possible de représenter les trois clusters distincts sur le même plan des deux premiers
        #composants de l'ACP en question
        plt.figure(figsize=(18, 6))
        plt.scatter(new_columns[:, 0], new_columns[:, 1])
        colors = ['black','red','green',"Blue"]
        for label, x, y,c in zip(df.index.values, new_columns[:, 0], new_columns[:, 1],df['cluster']):
            plt.annotate(
                label,
                xy=(x, y), xytext=(-5, 5),
                textcoords='offset points', ha='right', va='bottom',
                bbox=dict(boxstyle='round,pad=0.5', fc=colors[c], alpha=0.5),
                arrowprops = dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        plt.title("ACP rapprochant des departements similaires")
        
        st.pyplot(plt)
        
        st.write("<div style='text-align: justify'><p>On remarque alors que les clusters sont assez distincts et ne se croisent pas ce qui peut nous laisser penser que le nombre de clusters a été correctement choisi.</div>", unsafe_allow_html=True)
        
        st.write("<div style='text-align: justify'><p>Dans la dernière partie de cette étape de clustering, il d'interpréter les clusters. En effet, l'interprétation des clusters nous permettra d'établir les spécificités des différents sous-groupes et ainsi pouvoir les différencier selon des caractéristiques spécifiques.</div>", unsafe_allow_html=True)
        st.write("<div style='text-align: justify'><p>Pour faire cela, on va construire un cercle des corrélations. En effet, l'introduction de cette représentation nous permettra d'identifier les variables ayant une influence sur la position des actions dans le graphique précédent. Le cercle des corrélations est représenté ci-dessous</div>", unsafe_allow_html=True)
        
        n,p = X_normalise.shape #détermination du nombre d'observations et de colonnes
        eigval = ((n-1) / n) * acp.explained_variance_ # détermination des valeurs propres
        sqrt_eigval = np.sqrt(eigval) # détermination de la racine carrée des valeurs propres
        corvar = np.zeros((p,p)) # définition de la matrice vide
        for k in range(p):
            corvar[:,k] = acp.components_[k,:] * sqrt_eigval[k]

        #formation du dataframe regroupant l'ensemble de ces informations
        df_coord = pd.DataFrame({'id': test, 'corr_axe1': corvar[:,0], 'corr_axe2': corvar[:,1]})
        
        correlations_axe1 = []
        correlations_axe2 = []

        for i,feat in enumerate(test):
            corr_axe1 = np.corrcoef(X_normalise[:,i],df_new['pca_1'].values)[0][1]
            corr_axe2 = np.corrcoef(X_normalise[:,i],df_new['pca_2'].values)[0][1]
            correlations_axe1.append(corr_axe1)
            correlations_axe2.append(corr_axe2)
    
        df_coord = pd.DataFrame({'id': test, 'corr_axe1': correlations_axe1, 'corr_axe2': correlations_axe2})
        
        #Création du cercle de corrélation pour les deux premiers composants
        fig, ax = plt.subplots(figsize=(10, 9))
        texts = []
        for i in range(0, acp.components_.shape[1]):
            ax.arrow(0,
                     0,  # Start the arrow at the origin
                     acp.components_[0, i]*sqrt_eigval[0],  #0 for PC1
                     acp.components_[1, i]*sqrt_eigval[1],  #1 for PC2
                     head_width=0.01,
                     head_length=0.01, 
                     width=0.001,              )
    
            random_position_x = random.choice(np.arange(0,0.1,0.02))
            random_position_y = random.choice(np.arange(0,0.1,0.02))
            plt.text(acp.components_[0, i]*sqrt_eigval[0]+random_position_x,acp.components_[1, i]*sqrt_eigval[1]+random_position_y,test[i])


        plt.plot([-1, 1], [0, 0], color='grey', ls='--')
        plt.plot([0, 0], [-1, 1], color='grey', ls='--')

        plt.xlabel('PC{} ({}%)'.format(1, round(100*acp.explained_variance_ratio_[0],1)))
        plt.ylabel('PC{} ({}%)'.format(2, round(100*acp.explained_variance_ratio_[1],1)))

        plt.title("Cercle des corrélations (PC{} et PC{})".format(1, 2))


        an = np.linspace(0, 2 * np.pi, 100)
        plt.plot(np.cos(an), np.sin(an)) 
        plt.axis('equal')
        st.pyplot(plt)
        
        st.write("<div style='text-align: justify'><p>Basé sur ce graphique, on peut donc réaliser une première interprétation des différents clusters de notre base de données. Sur le cercle de corrélations, on peut voir que les flèches des variables de rendements \"2019-11-30\", \"2019-06-30\" et \"2021-12-31\" semblent pointer vers le haut de façon presque parallèle avec l'axe vertical du graphique. Cela nous indique donc que sur le graphique associé à l'ACP, les actions sont répartis de haut en bas principalement selon les valeurs associés à ces variables.Ainsi, les actions ayant les rendements les plus importants sur ces trois périodes mensuelles se trouveront dans la partie supérieure du graphique tandis que le reste se trouveront dans la partie inférieure de celui-ci.</div>", unsafe_allow_html=True)

        st.write("<div style='text-align: justify'><p>Par conséquent, lorsqu'on regarde ce même graphique, on peut remarquer que les actions du cluster noir se retrouvent dans la partie supérieure du graphique tandis que celles des clusters rouges et verts se trouvent dans la partie médiane-basse du plan. Par conséquent, on peut réaliser une première interprétation des clusters nouvellement formés : les actions du cluster noirs ont comme spécificité d'avoir des rendements plus élevés que la moyenne des autres clusters pour les dates du 30 juin 2019, 30 novembre 2019 et 31 décembre 2021</div>", unsafe_allow_html=True)

        
        
            
    else :
        st.header("CAH : Classification Ascendante Hiérarchique")
        
        st.write("<div style='text-align: justify'><p>Une fois les valeurs manquantes imputées et les données normalisées de la même façon qu'au préalable, l'algorithme de clustering hierarchique est implémenté. Etant donné que les données sont des valeurs quantitatives continues, l'implémentation de la méthode Ward et de la distance euclidienne ont été privilégié</div>", unsafe_allow_html=True) 
                 
        st.write("<div style='text-align: justify'><p>Les résultats de la répartition des actions dans les différents clusters, sous la forme d'un dendrogramme, sont représentés ci-dessous :</div>", unsafe_allow_html=True)
        
        #import des librairies classiques
        import numpy as np 
        import pandas as pd 
        import matplotlib.pyplot as plt 
        import seaborn as sns

        roic = pd.read_excel(r"./FinanceQuant/Data_projet_nettoye.xlsx",sheet_name='ROIC') #import de la base des données ROIC
        roic = roic.drop([0,1]) #suppression des premières lignes vides
        roic = roic.set_index('Stock') #on met les dates en index
        
        roic = roic.drop(['VNOM UW Equity','BKR US Equity','NTR CT Equity','NTR US Equity','GLEN LN Equity','CTVA US Equity','DOW US Equity','RIO US Equity','MNDI LN Equity','AAL LN Equity','BHP US Equity','PXD US Equity'],axis=1)

        roic = roic.bfill(axis ='rows')
        roic = roic.ffill(axis='rows')
        
        roic_t = roic.T #transposition du dataframe pour avoir les actions en lignes
        roic_t_columns = roic_t.columns.to_list()
        X = roic_t[roic_t_columns].values
        
        #normalisation des données, nécessaire à l'implémentation d'un clustering hierarchique
        from sklearn.preprocessing import MinMaxScaler
        sc = MinMaxScaler()
        X_normalise = sc.fit_transform(X)
        
        #import de la librairie permettant la mise en place du clustering hierarchique
        from scipy.cluster.hierarchy import linkage,dendrogram,fcluster

        #implémentation des paramètres du clustering hierarchique : méthode ward et calcul de la distance 
        #euclidienne
        matrice_distance = linkage(X,metric='euclidean',method='ward')

        #création du dendrogramme associé au clustering hierarchique permettant de définir les différents clusters
        plt.title("Distance entre les actions (CAH) ")
        dendrogram(matrice_distance,orientation="right",leaf_font_size=8,
          color_threshold = 150, labels = roic_t.index.str.strip().values
          )
        st.pyplot(plt)
        
        st.write("On remarque alors que 3 clusters se distinguent lors de la construction du dendrogramme associé à l'algorithme de clustering hierarchique implémenté.} Par conséquent, on les distingue à l'aide de trois différentes couleurs sur le graphique associé. Si on souhaite réaliser une comparaison avec les clusters définis avec l'algorithme K-means, on peut dire que les trois sous-groupes obtenus à l'aide du clustering hierarchique sont moins homogènes. Cependant, ce résultat peut être considéré comme biaisé, étant donné que beaucoup d'actions ont du être supprimé à cause du nombre important de valeurs manquantes.")
        
        

        


            
    
elif menu == "Moyenne-Variance":
    sub_menu_moyenne_variance = st.sidebar.radio("Sous-section Moyenne-Variance", ["Représentation des actifs sur un plan rendement/volatilité", "Représentation des portefeuilles sur un plan rendement/volatilité", "Portefeuille à variance minimale", "Portefeuille équipondérant"])
    if sub_menu_moyenne_variance == "Représentation des actifs sur un plan rendement/volatilité":
        st.header("Représentation des actifs sur un plan rendement/volatilité")
        
        st.write("<div style='text-align: justify'><p>Dans cette partie, nous allons réaliser une analyse moyenne-variance vis-à-vis de cette base de données. De la même façon que pour le clustering, l'analyse détaillée de cette partie est disponible dans le rapport</div>", unsafe_allow_html=True)
                 
        st.write("<div style='text-align: justify'><p>On va d'abord commencer par représenter les actions sur un plan rendement-volatilité. En plus de ces actions, on représentera également la frontière d'efficience, correspondant à l'ensemble des portefeuilles pouvant être créés à partir des actions de la base de données maximisant les rendements, pour chaque niveau de risque.</div>", unsafe_allow_html=True)
                 
        pivot_stock_return = pd.read_excel(r"./FinanceQuant/Data_projet_finance.xlsx',sheet_name='Returns")
        pivot_stock_return = pivot_stock_return.T
        pivot_stock_return = pivot_stock_return.reset_index()
        pivot_stock_return.columns = pivot_stock_return.iloc[0]
        pivot_stock_return = pivot_stock_return.drop([pivot_stock_return.index[0]])
        
        pivot_stock_return = pivot_stock_return.bfill(axis='rows')
        pivot_stock_return = pivot_stock_return.ffill(axis='rows')
        
        pivot_stock_return.reset_index(drop=True, inplace=True)
        
        pivot_stock_return['Stock'] = pd.to_datetime(pivot_stock_return['Stock'])
        pivot_stock_return = pivot_stock_return.set_index('Stock')
        
        pivot_stock_return.drop(columns=['S5ENRS Index'], axis=1, inplace=True)
        
        import matplotlib.pyplot as plt
        mu = pivot_stock_return.mean(axis=0)
        s = pivot_stock_return.cov()

        from pypfopt.efficient_frontier import EfficientFrontier
        
        fig, ax = plt.subplots()
        ef = EfficientFrontier(mu, s)
        plotting.plot_efficient_frontier(ef, ax=ax, show_assets=True)
        
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
                 
        st.write("<div style='text-align: justify'><p>On remarque que la majorité des actions sont regroupés dans la partie inférieure-gauche du graphique. Cependant, une action se distingue des autres, en haut à droite du graphique (APA US Equity) avec un niveau de rendement et de risque beaucoup plus important que les autres sur la période.</div>", unsafe_allow_html=True)
                 
    
    elif sub_menu_moyenne_variance == "Représentation des portefeuilles sur un plan rendement/volatilité" :
        st.header("Représentation des portefeuilles sur un plan rendement/volatilité")

        st.write("<div style='text-align: justify'><p>Ensuite, il sera intéressant de représenter l'ensemble des portefeuilles potentiellement construisables à partir de ces actions sur ce même graphique. C'est ce que nous allons réaliser dans le graphique ci-dessous. Dans le cadre de la réalisation de ce graphique sous Python, une limite de 10 000 portefeuilles représentable est fixée. Cependant, une autre représentation de l'ensemble des portefeuilles pourra être retrouvé dans le rapport final associé à cette application</div>", unsafe_allow_html=True)
        
        
        pivot_stock_return = pd.read_excel(r"./FinanceQuant/Data_projet_finance.xlsx",sheet_name='Returns')
        pivot_stock_return = pivot_stock_return.T
        pivot_stock_return = pivot_stock_return.reset_index()
        pivot_stock_return.columns = pivot_stock_return.iloc[0]
        pivot_stock_return = pivot_stock_return.drop([pivot_stock_return.index[0]])
        
        pivot_stock_return = pivot_stock_return.bfill(axis='rows')
        pivot_stock_return = pivot_stock_return.ffill(axis='rows')
        
        pivot_stock_return.reset_index(drop=True, inplace=True)
        
        pivot_stock_return['Stock'] = pd.to_datetime(pivot_stock_return['Stock'])
        pivot_stock_return = pivot_stock_return.set_index('Stock')
        
        pivot_stock_return.drop(columns=['S5ENRS Index'], axis=1, inplace=True)
        
        import matplotlib.pyplot as plt
        mu = pivot_stock_return.mean(axis=0)
        s = pivot_stock_return.cov()
        
        fig1, ax = plt.subplots()
        ef = EfficientFrontier(mu, s)
        plotting.plot_efficient_frontier(ef, ax=ax, show_assets=True)

        import numpy as np
        n_samples = 10000
        w = np.random.dirichlet(np.ones(ef.n_assets), n_samples)
        rets = w.dot(ef.expected_returns)
        stds = np.sqrt(np.diag(w @ ef.cov_matrix @ w.T))
        sharpes = rets / stds
        ax.scatter(stds, rets, marker=".", c=sharpes, cmap="viridis_r")

        ax.legend()
        plt.tight_layout()
        st.pyplot(fig1)
    
    elif sub_menu_moyenne_variance == "Portefeuille à variance minimale" :
        st.header("Représentation du portefeuille à variance minimale")
        
        st.write("<div style='text-align: justify'><p>Puis, nous allons chercher à analyser des portefeuilles caractéristiques de cette base de données. Tout d'abord, nous allons commencer par analyser le portefeuille minimisant la volatilité, le portefeuille le moins risqué</div>", unsafe_allow_html=True)
        
        st.write("<div style='text-align: justify'><p>Nous commencerons par le représenter dans un plan rendement-volatilité</div>", unsafe_allow_html=True)
        
        pivot_stock_return = pd.read_excel(r"./FinanceQuant/Data_projet_finance.xlsx",sheet_name='Returns')
        pivot_stock_return = pivot_stock_return.T
        pivot_stock_return = pivot_stock_return.reset_index()
        pivot_stock_return.columns = pivot_stock_return.iloc[0]
        pivot_stock_return = pivot_stock_return.drop([pivot_stock_return.index[0]])
        
        pivot_stock_return = pivot_stock_return.bfill(axis='rows')
        pivot_stock_return = pivot_stock_return.ffill(axis='rows')
        
        pivot_stock_return.reset_index(drop=True, inplace=True)
        
        pivot_stock_return['Stock'] = pd.to_datetime(pivot_stock_return['Stock'])
        pivot_stock_return = pivot_stock_return.set_index('Stock')
        
        pivot_stock_return.drop(columns=['S5ENRS Index'], axis=1, inplace=True)
        
        import matplotlib.pyplot as plt
        mu = pivot_stock_return.mean(axis=0)
        s = pivot_stock_return.cov()
        
        ef = EfficientFrontier(mu, s)
        
        import json
        ef1 = EfficientFrontier(mu, s, weight_bounds=(0, 1))
        ef1.min_volatility()
        weights_min_volatility = ef1.clean_weights()
        test = ef1.portfolio_performance(verbose=True)
        print(f'Portfolio weights for min volatility optimisation (lowest level of risk): {json.dumps(weights_min_volatility, indent=4, sort_keys=True)} \n')
        print(f'Portfolio performance: {ef1.portfolio_performance()} \n')
        
        import matplotlib.pyplot as plt
        fig2, ax = plt.subplots()
        ef = EfficientFrontier(mu, s)
        plotting.plot_efficient_frontier(ef, ax=ax, show_assets=True)

# Find the tangency portfolio
        ret_tangent, std_tangent, _ = test
        ax.scatter(std_tangent, ret_tangent, marker="o", s=100, c="r", label="Minimum variance portfolio")

        ax.legend()
        plt.tight_layout()
        st.pyplot(fig2)
        
        st.write("<div style='text-align: justify'><p>Puis, nous allons analyser les caractéristiques de ce portefeuille. On va commencer par présenter les différents poids de chacune des 9 actions ayant permis de composer ce portefeuille</div>", unsafe_allow_html=True)
        
        st.markdown("**Poids du portefeuille** : ")

        
        for stock, weight in weights_min_volatility.items():
            st.write(f"- {stock}: {weight}")
            
        
        st.write("<div style='text-align: justify'><p>On peut remarquer une répartition plutôt équilibrée entre les 9 actions de ce portefeuille, avec les poids les plus importants concernant les actions \"NEM US Equity\" et \"UPM FH Equity\"</div>", unsafe_allow_html=True)
        
        st.markdown("**Performances du portefeuille** : ")
        st.write("Rendement annuel attendu : ",  round(ef1.portfolio_performance(verbose=True, risk_free_rate=0.01305)[0],2))
        st.write("Volatilité annuelle: ",  round(ef1.portfolio_performance(verbose=True, risk_free_rate=0.01305)[1],2))
        st.write("Ratio de Sharpe: ",  round(ef1.portfolio_performance(verbose=True, risk_free_rate=0.01305)[2], 2))
        
        st.write("<div style='text-align: justify'><p>Le rendement de ce portefeuille est de 0,503 sur la période tandis que la volatilité est de 4,37 ce qui reste élevé pour un portefeuille ayant comme objectif de minimiser la variance. Cependant, cela peut s'expliquer par le fait que les données étant mensuelles, elles fluctuent très fortement d'un mois à l'autre. Le ratio de Sharpe est alors égal à 0,11.</div>", unsafe_allow_html=True)
        
    else : 

        st.header("Représentation du portefeuille équiponderant")
        st.write("<div style='text-align: justify'><p>Enfin, nous allons construire un portefeuille équipondérant. Un portefeuille équipondérant est un portefeuille composé de toutes les actions disponibles avec un poids égal. Dans le cadre de notre projet, cela correspond à l'établissement d'un portefeuille contenant les 58 actions de la base de données avec un poids similaire équivalent à 1/58 soit environ 1,72%. Ainsi, si un investisseur souhaitait investir une somme de 1000$, il faudrait investir 17,2 dollars dans chacune des 58 stocks pour constituer un portefeuille équipondéré regroupant l'ensemble des actions.</div>", unsafe_allow_html=True)
        
        pivot_stock_return = pd.read_excel(r"./FinanceQuant/Data_projet_finance.xlsx",sheet_name='Returns')
        pivot_stock_return = pivot_stock_return.T
        pivot_stock_return = pivot_stock_return.reset_index()
        pivot_stock_return.columns = pivot_stock_return.iloc[0]
        pivot_stock_return = pivot_stock_return.drop([pivot_stock_return.index[0]])
        
        pivot_stock_return = pivot_stock_return.bfill(axis='rows')
        pivot_stock_return = pivot_stock_return.ffill(axis='rows')
        
        pivot_stock_return.reset_index(drop=True, inplace=True)
        
        pivot_stock_return['Stock'] = pd.to_datetime(pivot_stock_return['Stock'])
        pivot_stock_return = pivot_stock_return.set_index('Stock')
        
        pivot_stock_return.drop(columns=['S5ENRS Index'], axis=1, inplace=True)
        
        import matplotlib.pyplot as plt
        mu = pivot_stock_return.mean(axis=0)
        s = pivot_stock_return.cov()
        
        ef1 = EfficientFrontier(mu, s, weight_bounds=(0, 1))
        ef1.min_volatility()
        weights_min_volatility = ef1.clean_weights()
        
        from collections import OrderedDict

        for i in weights_min_volatility:
            weights_min_volatility[i] = 1/58
        #on change les poids pour qu'ils soient équipondérants sur toutes les actions avec une somme à 1
    
        test = weights_min_volatility
        ef1.set_weights(test) #on applique cette modification au portefeuille
        equal_weight = ef1.clean_weights()

        st.markdown("**Poids du portefeuille** : ")
        for stock, weight in equal_weight.items():
            st.write(f"- {stock}: {weight}")
    
        st.markdown("**Performances du portefeuille** : ")
        performance = ef1.portfolio_performance(verbose=True)
        st.write("Rendement annuel attendu : ", round(performance[0],2))
        st.write("Volatilité annuelle: ", round(performance[1], 2))
        st.write("Ratio de Sharpe: ", round(performance[2], 2))
        
        st.write("<div style='text-align: justify'><p>Le rendement de ce portefeuille équipondéré est alors de 1.32, assez logiquement en hausse par rapport au précédent portefeuille. Cependant, il est aussi plus risqué avec une volatilité égale à 9,38. Le ratio de Sharpe est alors égale à 0,14.</div>", unsafe_allow_html=True)
    

    
else:
    st.header("Portfolio construction and monitoring")
    sub_menu = st.sidebar.radio("Sous-section Portfolio construction and monitoring",
                                ["Construction du portefeuille", "Monitoring du portefeuille"])
    if sub_menu == "Construction du portefeuille":
    
        st.write("Nous allons vous présenter dans cette dernière partie un portefeuille adapté à une personne voulant un bon équilibre entre rendements et risque tout en gardant une certaine stabilité en cas de chocs de par l'intégration d'actifs ayant de bons scores environnementaux, sociaux et de gouvernance (ESG).")
        st.write(" En effet, nous avons fait le choix de bonifier l'allocation des actifs ayant un bon score ESG et défavoriser  ceux en ayant un mauvais de sorte que les actifs ayant un bon score ESG ait plus de poids dans le portefeuille proposé.")

        st.write("Votre portefeuille sera constitué des actifs suivant : ")
        st.image("portfolio_weights.png")

        st.write(" Ce sont des actifs appartenant pour la plupart au secteur de l'énergie et des matériaux.")

    else:
        st.write(" Afin d'adapter votre portefeuille aux variations du marché, nous avons tenter de les identifier en amont à l'aide du croisement de moyennes mobiles qui nous permet de récolter des signaux d'achat ou de vente pour chaque actif contenu dans le portefeuille.")
        st.write(" En effet, lorsque une moyenne mobile de court terme croise une moyenne mobile de long terme et dépasse celle-ci alors un signal d'achat est généré."
                 "Notre signal vous permettra d'acheter et de vendre vos actifs à point nommé avant que la tendance ne s'inverse.")

        st.write("Par exemple, on constate à travers l'image ci-dessous que pour l'action HES US Equity, il est judicieux de l'acheter en 2022, quelque soit la période.")
        st.image("HES_US_Equity.jpeg")
