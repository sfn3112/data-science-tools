

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import plotly.graph_objects as go


class PC_Analysis:
    """A complete principal component analysis"""

    def __init__(self, data, parm_n=None):
        """initiate the PCA model and the data"""
        self.X_std = StandardScaler().fit_transform(data)
        self.model = PCA(parm_n).fit(self.X_std)
        self.variables_name = list(data.columns)
        self.individuals_name = list(data.index)

    def individuals_coor_PC(self):
        """fit the model with X and apply the dimensionality reduction on X_std
        returns:
        -------
               [array, shape(n_samples, n_components)] transformed values in the PC planes
        """
        return self.model.fit_transform(self.X_std)

    def plot_explained_variance(self):
        """ Plot the explained variance of PCA model
        returns:
        -------
            fig: [plotly.graph_objs._figure.Figure] a plotly figure object
        """

        y_bar = self.model.explained_variance_ratio_
        y_scatter = np.cumsum(self.model.explained_variance_ratio_)
        x = ["PC_" + str(i) for i in range(1, len(y_bar) + 1)]

        data_Bar = go.Bar(x=x, y=y_bar, name='Individual')
        data_Scatter = go.Scatter(x=x, y=y_scatter, name='Cumulative', mode='lines+markers')

        fig = go.Figure([data_Bar, data_Scatter])

        fig.update_layout(
            title={'text': "Explained variance by principale components",
                   'y': 0.90,
                   'x': 0.5,
                   'font': {'family': "Arial",
                            'size': 23,
                            'color': "black"}},
            yaxis={'title': 'Explained variance %'},
            xaxis={'title': 'Principale components'},
            font={'family': "Arial",
                  'size': 15,
                  'color': "black"},
            showlegend=True)

        return fig

    def correlation_circle(self, PC_a, PC_b, ratio_a=None, ratio_b=None):
        """ Correlation between the variables and the principal components
        Parameters:
        -----------
            PC_a: [int]  order of the principal component ( x-axis in the factor-plane)
            PC_b: [int]  order of the principal component ( y-axis in the factor-plane)
            ratio_a: [float] threshold to plot the variables on the x-axis of the correlation circle
            ratio_b: [float] threshold to plot the variables on the x-axis of the correlation circle
        returns:
        -------
            corrOldNew: [pd.DataFrame] the correlation table between variables and the principal components
            fig: [plotly.graph_objs._figure.Figure] correlation circle
        """

        PC_a -= 1
        PC_b -= 1

        individuals_PC = self.individuals_coor_PC()

        # compute the correlation between old features and the principal compoenents
        tmp = np.corrcoef(self.X_std.T, individuals_PC.T)
        corrOldNew = tmp[0:len(self.variables_name), len(self.variables_name):]

        corrOldNew = pd.DataFrame(corrOldNew, columns=['PC_' + str(i) for i in range(1, corrOldNew.shape[1] + 1)],
                                  index=self.variables_name)

        del tmp

        if not (ratio_a is None) and not (ratio_b is None):
            corrOldNew_ = corrOldNew.loc[
                          (corrOldNew.abs().iloc[:, PC_a] > ratio_a) & (corrOldNew.abs().iloc[:, PC_b] > ratio_b), :]
        elif not (ratio_a is None):
            corrOldNew_ = corrOldNew.loc[(corrOldNew.abs().iloc[:, PC_a] > ratio_a), :]
        elif not (ratio_b is None):
            corrOldNew_ = corrOldNew.loc[(corrOldNew.abs().iloc[:, PC_b] > ratio_b), :]
        else:
            corrOldNew_ = corrOldNew

        data = []

        for i_feature, variable_name in enumerate(list(corrOldNew_.index)):
            # corr_coeff_x = corrOldNew[i_feature][PC_a]
            # corr_coeff_y = corrOldNew[i_feature][PC_b]
            corr_coeff_x = corrOldNew_.iloc[i_feature, PC_a]
            corr_coeff_y = corrOldNew_.iloc[i_feature, PC_b]
            trace = go.Scatter(x=[0, corr_coeff_x], y=[0, corr_coeff_y], mode='lines+text',
                               text=['', variable_name], textposition="top right", name=variable_name)
            data.append(trace)

        fig = go.Figure(data)

        fig.update_layout(
            title={
                'text': "Projection of variables on the factor plane [" + str(PC_a + 1) + " X " + str(PC_b + 1) + "]",
                'y': 0.95,
                'x': 0.45,
                'font': {'family': "Arial",
                         'size': 23,
                         'color': "black"}},
            width=880,
            height=800,
            shapes=[{'type': "circle", 'xref': "x", 'yref': "y", 'x0': -1, 'y0': -1, 'x1': 1, 'y1': 1,
                     'line_color': "LightSeaGreen"},
                    {'type': "line", 'x0': -1, 'y0': 0, 'x1': 1, 'y1': 0},
                    {'type': "line", 'x0': 0, 'y0': -1, 'x1': 0, 'y1': 1}],
            xaxis=dict(title='PC' + str(PC_a + 1), showline=False),
            yaxis=dict(title='PC' + str(PC_b + 1), showline=False))

        return corrOldNew, fig

    def individuals_quality_representation(self):
        """ Quality of representation of individuals on the principal components
        returns:
        --------
            quality: [pd.DataFrame, shape(n_individuals, n_components)]
        """
        individuals_PC = self.individuals_coor_PC()

        qual = individuals_PC * individuals_PC
        qual = (qual.T / qual.sum(axis=1)).T
        quality = pd.DataFrame(data=qual * 100, index=self.individuals_name,
                               columns=list(range(1, self.model.n_features_ + 1)))
        del qual
        quality.columns = ['CP_' + str(col) for col in quality.columns]
        return quality

    def individuals_contribution(self):
        """ Contribution of individuals in building the principal components
        returns:
        --------
            contribution: [pd.DataFrame, shape(n_individuals, n_components)]
        """
        individuals_PC = self.individuals_coor_PC()
        contr = individuals_PC * individuals_PC
        contr = contr / contr.sum(axis=0)
        contribution = pd.DataFrame(data=contr * 100, index=self.individuals_name,
                                    columns=list(range(1, self.model.n_features_ + 1)))
        del contr
        contribution.columns = ['CP_' + str(col) for col in contribution.columns]
        return contribution

    def projection_PC_space(self, PC_a, PC_b, target, quality_ratio_a=None, quality_ratio_b=None):
        """ Projection of individials in the factor-plane [PC_a, PC_b] with the respected quality of representation
        quality_ratio_a and quality_ratio_b.
        parameters:
        -----------
            PC_a: [int]  order of the principal component ( x-axis in the factor-plane)
            PC_b: [int]  order of the principal component ( y-axis in the factor-plane)
            quality_ratio_a: [float] quality representation threshold to plot the the individuals on the x-axis of the factor-plane
            quality_ratio_b: [float] quality representation threshold to plot the the individuals on the y-axis of the factor-plane
        returns:
        --------
            fig: [plotly.graph_objs._figure.Figure] individuals projection on the factor-plane
        """
        PC_a -= 1
        PC_b -= 1

        individuals_PC = self.individuals_coor_PC()

        # individuals quality
        individuals_quality = self.individuals_quality_representation()

        if not (quality_ratio_a is None) and not (quality_ratio_b is None):
            bool_index = (individuals_quality.iloc[:, PC_a] > quality_ratio_a) & (
                        individuals_quality.iloc[:, PC_b] > quality_ratio_b)
            target = target.loc[bool_index]
            individuals_PC = individuals_PC[bool_index.to_numpy().reshape(-1), :]
            individuals_name_ = pd.Series(self.individuals_name).loc[bool_index]
        elif not (quality_ratio_a is None):
            bool_index = (individuals_quality.iloc[:, PC_a] > quality_ratio_a)
            target = target.loc[bool_index]
            individuals_PC = individuals_PC[bool_index.to_numpy().reshape(-1), :]
            individuals_name_ = pd.Series(self.individuals_name).loc[bool_index]
        elif not (quality_ratio_b is None):
            bool_index = (individuals_quality.iloc[:, PC_b] > quality_ratio_b)
            target = target.loc[bool_index]
            individuals_PC = individuals_PC[bool_index.to_numpy().reshape(-1), :]
            individuals_name_ = pd.Series(self.individuals_name).loc[bool_index]
        else:
            individuals_name_ = pd.Series(self.individuals_name)

        del individuals_quality

        target_1 = (target == 1).to_numpy().reshape(-1)
        target_0 = (target == 0).to_numpy().reshape(-1)

        trace_1 = go.Scatter(x=individuals_PC[target_1, PC_a], y=individuals_PC[target_1, PC_b], mode='markers',
                             marker={'size': 4}, name="y_1", hovertext=individuals_name_[target_1], opacity=0.7)
        trace_0 = go.Scatter(x=individuals_PC[target_0, PC_a], y=individuals_PC[target_0, PC_b], mode='markers',
                             marker={'size': 4}, name="y_0", hovertext=individuals_name_[target_0], opacity=0.7)
        fig = go.Figure(data=[trace_1, trace_0])

        fig.update_layout(
            title={
                'text': "Projection of individuals on the factor plane [" + str(PC_a + 1) + " X " + str(PC_b + 1) + "]",
                'y': 0.90,
                'x': 0.5,
                'font': {'family': "Arial",
                         'size': 18,
                         'color': "black"}},
            xaxis=dict(title='PC' + str(PC_a + 1), showline=True),
            yaxis=dict(title='PC' + str(PC_b + 1), showline=True))

        # fig.show(config={'staticPlot':True})
        return fig
