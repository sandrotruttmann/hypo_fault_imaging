#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HYPOCENTER-BASED 3D IMAGING OF ACTIVE FAULTS: Visualisation Module

Please cite: Truttmann et al. (2023). Hypocenter-based 3D Imaging of Active Faults: Method and Applications in the Southwestern Swiss Alps.

@author: Sandro Truttmann
@contact: sandro.truttmann@gmail.com
@license: GPL-3.0
@date: April 2023
@version: 0.1.1
"""

# Import the needed modules
import numpy as np
import pandas as pd
import datetime
import plotly.graph_objects as go
import math
import mplstereonet
import utilities
import utilities_plot
import matplotlib.pyplot as plt
import os


def model_3d(input_params, data_input, data_input_outliers, data_output):
    """
    Generate an interactive 3D model with plotly.

    Parameters
    ----------
    input_params : DataFrame
        Input parameters.
    data_input : DataFrame
        Input data.
    data_input_outliers : DataFrame
        Input data that was identified as outliers.
    data_output : DataFrame
        Output data.

    Returns
    -------
    Interactive 3D model, saved in the output directory.

    """
    print('')
    print('Creating visual output...')
    
    # Unpack input parameters from dictionary
    for key, value in input_params.items():
        globals()[key] = value

    df = pd.merge(data_input, data_output, on='ID')

    fig = go.Figure()

    ############################################################################
    # Plot hypocenters
    df['Date'] = pd.to_datetime(df['Date'])
    min_date = df['Date'].min()
    color_date = df['Date'].apply(lambda x: (x - min_date).days)
    tick_interval = 365
    max_days = color_date.max()
    colticks = np.arange(0, (int(max_days / 1) + 1) * 1, tick_interval)
    coldatetimes = [min_date + datetime.timedelta(days=i)
                    for i in colticks.tolist()]
    coltext = [i.strftime("%d-%b-%Y") for i in coldatetimes]
    trace = go.Scatter3d(
        x=df['_X'],
        y=df['_Y'],
        z=df['_Z'],
        mode='markers',
        marker=dict(
            color=color_date,
            colorscale='Rainbow',
            colorbar=dict(
                title='Date',
                tickvals=colticks,
                ticktext=coltext,
                xanchor='left',
                x=0
                ),
            size=3,
            showscale=True),
        customdata=data_output,
        hovertemplate=
            '<b>Event ID:</b> %{customdata[0]} <br>'
            '<b>Class:</b> %{customdata[20]} <br>'
            '<br>'
            '<b>Fault parameters:</b> <br>'
            '<b>Rupture radius (m):</b> %{customdata[2]:.0f} <br>'
            '<b>Fault Plane Orientation:</b> %{customdata[12]} / %{customdata[13]} <br>'
            # '<b>κ:</b> %{customdata[10]:.0f} <br>'
            '<b>kappa:</b> %{customdata[10]:.0f} <br>'
            # '<b>β:</b> %{customdata[11]:.0f} <br>'
            '<b>beta:</b> %{customdata[11]:.0f} <br>'
            '<br>'
            '<b>Stress parameters:</b> <br>'
            '<b>Rake:</b> %{customdata[17]:.0f} <br>',
        legendgroup='hypocenter',
        name='Relocated hypocenters',
        showlegend=True,
        # visible='legendonly',
        visible=True,
        )
    fig.add_traces(trace)
    
    # Plot outliers
    # Plot hypocenters
    # data_input_outliers['Date'] = pd.to_datetime(df['Date'])
    # min_date = data_input_outliers['Date'].min()
    # color_date = data_input_outliers['Date'].apply(lambda x: (x - min_date).days)
    # tick_interval = 365
    # max_days = color_date.max()
    # colticks = np.arange(0, (int(max_days / 1) + 1) * 1, tick_interval)
    # coldatetimes = [min_date + datetime.timedelta(days=i)
    #                 for i in colticks.tolist()]
    # coltext = [i.strftime("%d-%b-%Y") for i in coldatetimes]
    trace = go.Scatter3d(
        x=data_input_outliers['_X'],
        y=data_input_outliers['_Y'],
        z=data_input_outliers['_Z'],
        mode='markers',
        marker=dict(
            # color=color_date,
            color='black',
            opacity=0.5,
            colorscale='Rainbow',
            colorbar=dict(
                title='Date',
                tickvals=colticks,
                ticktext=coltext,
                xanchor='left',
                x=0
                ),
            size=3,
            showscale=True),
        legendgroup='hypocenter_outliers',
        name='Relocated hypocenters (outliers)',
        showlegend=True,
        visible=True,
        )
    fig.add_traces(trace)

    ############################################################################
    # Plot error ellipsoids
    # Workaround to only show one legend entry for fault planes: create an array
    # with only the first value True with the length of the number of events
    idx = df['EX'].dropna().index[0]
    legend_show = [False for i in range(len(df))]
    legend_show[idx] = True

    for i in range(len(df)):
        # Create error ellipse at the zero point
        phi = np.linspace(0, 2 * np.pi, 10)
        theta = np.linspace(-np.pi / 2, np.pi / 2, 10)
        phi, theta = np.meshgrid(phi, theta)
        x = np.cos(theta) * np.sin(phi) * df['EX'][i] * 3
        y = np.cos(theta) * np.cos(phi) * df['EY'][i] * 3
        z = np.sin(theta) * df['EZ'][i] * 3
    
        # Shift error ellipse to the right xyz coordinates
        x = x + df['_X'][i]
        y = y + df['_Y'][i]
        z = z + df['_Z'][i]
    
        trace = go.Mesh3d(x=x.flatten(),
                          y=y.flatten(),
                          z=z.flatten(),
                          color='grey',
                          opacity=0.2,
                          alphahull=0,
                          hoverinfo='none',
                          showlegend=legend_show[i],
                          name='Error ellipsoids (3σ)',
                          legendgroup='Error ellipsoids (3σ)',
                          visible='legendonly')
        fig.add_trace(trace)
    
    # Plot error ellipsoids of outliers
    idx = data_input_outliers['EX'].dropna().index[0]
    legend_show = [False for i in range(len(data_input_outliers))]
    legend_show[idx] = True

    for i in range(len(data_input_outliers)):
        # Create error ellipse at the zero point
        phi = np.linspace(0, 2 * np.pi, 10)
        theta = np.linspace(-np.pi / 2, np.pi / 2, 10)
        phi, theta = np.meshgrid(phi, theta)
        x = np.cos(theta) * np.sin(phi) * data_input_outliers['EX'][i] * 3
        y = np.cos(theta) * np.cos(phi) * data_input_outliers['EY'][i] * 3
        z = np.sin(theta) * data_input_outliers['EZ'][i] * 3
    
        # Shift error ellipse to the right xyz coordinates
        x = x + data_input_outliers['_X'][i]
        y = y + data_input_outliers['_Y'][i]
        z = z + data_input_outliers['_Z'][i]
    
        trace = go.Mesh3d(x=x.flatten(),
                          y=y.flatten(),
                          z=z.flatten(),
                          color='grey',
                          opacity=0.2,
                          alphahull=0,
                          hoverinfo='none',
                          showlegend=legend_show[i],
                          name='Error ellipsoids (outliers) (3σ)',
                          legendgroup='Error ellipsoids (outliers) (3σ)',
                          visible='legendonly')
        fig.add_trace(trace)

    ############################################################################
    # Plot the focal planes
    if 'Strike1' in df.columns:
        # Colors for focal planes
        colormap = 'RdYlGn_r'
        column = np.array(df['epsilon'])
        minval = 0
        maxval = np.nanmax(column)
        maxval = math.ceil(maxval * 10) / 10.0
        colorsteps = 40
        colors = utilities_plot.colorscale(column, colormap, minval, maxval, colorsteps)
        
        # Workaround to only show one legend entry for fault planes: create an array
        # with only the first value True with the length of the number of events
        idx = df['Strike1'].dropna().index[0]
        legend_show = [False for i in range(len(df))]
        legend_show[idx] = True
    
        for i in range(len(df)):
            if pd.isnull(df['Strike1'][i]) is True:
                pass
            else:
                # Select the focal plane with the smaller angular difference to the
                # reconstructed fault plane
                if df['pref_foc'][i] == 1:
                    nor_x, nor_y, nor_z = utilities.plane_azidip_to_normal(df['Strike1'][i]
                                                            + 90 % 360,
                                                            df['Dip1'][i])
                    nor_pref = np.array([nor_x, nor_y, nor_z])
                    nor_x, nor_y, nor_z = utilities.plane_azidip_to_normal(df['Strike2'][i]
                                                            + 90 % 360,
                                                            df['Dip2'][i])
                    nor_nonpref = np.array([nor_x, nor_y, nor_z])
                    foc_color = 'black'
        
                elif df['pref_foc'][i] == 2:
                    nor_x, nor_y, nor_z = utilities.plane_azidip_to_normal(df['Strike2'][i]
                                                            + 90 % 360,
                                                            df['Dip2'][i])
                    nor_pref = np.array([nor_x, nor_y, nor_z])
                    nor_x, nor_y, nor_z = utilities.plane_azidip_to_normal(df['Strike1'][i]
                                                            + 90 % 360,
                                                            df['Dip1'][i])
                    nor_nonpref = np.array([nor_x, nor_y, nor_z])
                    foc_color = 'black'
                    
                else:
                    nor_x, nor_y, nor_z = utilities.plane_azidip_to_normal(df['Strike1'][i]
                                                            + 90 % 360,
                                                            df['Dip2'][i])
                    nor_pref = np.array([nor_x, nor_y, nor_z])
                    nor_x, nor_y, nor_z = utilities.plane_azidip_to_normal(df['Strike2'][i]
                                                            + 90 % 360,
                                                            df['Dip1'][i])
                    nor_nonpref = np.array([nor_x, nor_y, nor_z])
                    foc_color = 'lightgrey'
                    
                    
                # Get XYZ coordinates of the points of the circular fault plane
                x = df['_X'][i]
                y = df['_Y'][i]
                z = df['_Z'][i]
                p = [x, y, z]
                r = df['r'][i]
                X_pref, Y_pref, Z_pref = utilities_plot.circleplane(p, r, nor_pref)
                X_nonpref, Y_nonpref, Z_nonpref = utilities_plot.circleplane(p, r, nor_nonpref)
        
        
                # Preferred focal plane
                focals_pref = go.Scatter3d(
                    x=X_pref,
                    y=Y_pref,
                    z=Z_pref,
                    mode='lines',
                    line=dict(
                        color=colors[i],
                        # color=foc_color,
                        width=10),
                    hoverinfo='none',
                    legendgroup='Pref. focal planes',
                    name='Pref. focal planes',
                    showlegend=legend_show[i],
                    visible='legendonly'
                    )
                
                # Rake of preferred focal
                r = df['r'][i]
                if df['pref_foc'][i] == 1:
                    rake = df['Rake1'][i]
                    rake_color = 'black'
                elif df['pref_foc'][i] == 2:
                    rake = df['Rake2'][i]
                    rake_color = 'black'
                else:
                    rake = df['Rake1'][i]
                    rake_color = 'lightgrey'
                u, v, w = utilities_plot.slipvector_3D(p, r, nor_pref, rake)
                xx = [x + (x - u), u]
                yy = [y + (y - v), v]
                zz = [z + (z - w), w]
                trace = go.Scatter3d(x=xx, y=yy, z=zz,
                                      mode='lines',
                                      line=dict(
                                          color=rake_color,
                                          width=10),
                                      legendgroup='Pref. focal planes - Kinematics',
                                      hoverinfo='none',
                                      name='Pref. focal planes - Kinematics',
                                      showlegend=legend_show[i],
                                      visible='legendonly'
                                      )
                fig.add_trace(trace)

                # Non-preferred focal plane
                focals_nonpref = go.Scatter3d(
                    x=X_nonpref,
                    y=Y_nonpref,
                    z=Z_nonpref,
                    mode='lines',
                    line=dict(
                        color=foc_color,
                        width=5),
                    hoverinfo='none',
                    legendgroup='Non-pref. focal planes',
                    name='Non-pref. focal planes',
                    showlegend=legend_show[i],
                    visible='legendonly'
                    )
                fig.add_traces([focals_pref, focals_nonpref])

    ############################################################################
    # Plot the fault planes    
    if 'class' in df.columns:
        column = data_output['class']
        cmap = 'gnuplot'
        minval = np.nanmin(data_output['class']) - 1.1
        maxval = np.nanmax(data_output['class']) + 0.1
        colorsteps = 100
        colors = utilities_plot.colorscale(column, cmap, minval, maxval, colorsteps, cmap_reverse=False)
    else:
        colors = ['black'] * len(df)

    column = df['kappa']
    minval = 0
    # maxval = np.nanmax(column)/2
    maxval = 100000
    opac = utilities_plot.opacity(column, minval, maxval, 1000)

    # Workaround to only show one legend entry for fault planes: create an array
    # with only the first value True with the length of the number of events
    legend_show = [False for i in range(len(df))]
    idx = df['mean_azi'].dropna()
    try:
        idx = idx.index[0]   
        legend_show[idx] = True
    except IndexError:
        print("indexerror")
        pass
    
    for i in range(len(df)):
        # Get XYZ coordinates of the points of the circular fault plane around the
        # hypocenter (point p)
        p = [df['_X'][i],
              df['_Y'][i],
              df['_Z'][i]
              ]
        r = df['r'][i]
        nor = np.array([df['nor_x_mean'][i],
                        df['nor_y_mean'][i],
                        df['nor_z_mean'][i]])
        X, Y, Z = utilities_plot.circleplane(p, r, nor)

        faults = go.Scatter3d(
            x=X,
            y=Y,
            z=Z,
            mode='lines',
            line=dict(
                # !!!!!
                color=colors[i],
                # color='black',
                width=6),
            opacity=opac[i],
            hoverinfo='none',
            legendgroup='3D Fault Model',
            name='3D Fault Model',
            showlegend=legend_show[i],
            visible='legendonly',
            )
        fig.add_traces(faults)

    df_k = df[df['mean_dip'].isna()]
    df_k = df_k.reset_index(drop=True)
    trace = go.Scatter3d(
        x=df_k['_X'],
        y=df_k['_Y'],
        z=df_k['_Z'],
        mode='markers',
        marker=dict(
            color='rgb(0, 0, 0)',
            size=2,
            showscale=False),
        legendgroup='3D Fault Model',
        showlegend=False,
        visible='legendonly',
        )
    fig.add_traces(trace)

    ############################################################################
    # Plot stress states
    if 'I' in df.columns:
        # Plot the fault planes with fault instability
        colormap = 'plasma'
        column = np.array(df['I'])
        minval = 0
        # maxval = np.nanmax(column)
        maxval = 1
        colorsteps = 50
        colors = utilities_plot.colorscale(column, colormap, minval, maxval, colorsteps)
        
        for i in range(len(df)):
            # Get XYZ coordinates of the points of the circular fault plane around the
            # hypocenter (point p)
            p = [df['_X'][i],
                  df['_Y'][i],
                  df['_Z'][i]
                  ]
            r = df['r'][i]
            nor = np.array([df['nor_x_mean'][i],
                            df['nor_y_mean'][i],
                            df['nor_z_mean'][i]])
            X, Y, Z = utilities_plot.circleplane(p, r, nor)
        
            faults = go.Scatter3d(
                x=X,
                y=Y,
                z=Z,
                mode='lines',
                line=dict(
                    color=colors[i],
                    width=6),
                hoverinfo='none',
                legendgroup='Fault instability I',
                name='Fault instability I',
                showlegend=legend_show[i],
                visible='legendonly'
                )
            fig.add_traces(faults)
            
        df_k = df[df['mean_dip'].isna()]
        df_k = df_k.reset_index(drop=True)
        trace = go.Scatter3d(
            x=df_k['_X'],
            y=df_k['_Y'],
            z=df_k['_Z'],
            mode='markers',
            marker=dict(
                color='rgb(0, 0, 0)',
                size=2,
                showscale=False),
            legendgroup='Fault instability I',
            name='Fault instability I',
            hoverinfo='none',
            showlegend=False,
            visible='legendonly',
            )
        fig.add_traces(trace)

        
        # Plot the kinematics
        colormap = 'twilight_shifted'
        column = np.array(df['rake'])
        minval = -180
        maxval = 180
        colorsteps = 360
        colors = utilities_plot.colorscale(column, colormap, minval, maxval, colorsteps)
        
        for i in range(len(df)):
            if df['rake'][i] == np.nan:
                pass
            else:
                x = df['_X'][i]
                y = df['_Y'][i]
                z = df['_Z'][i]
                p = [x, y, z]
                r = df['r'][i]
                nor = np.array([df['nor_x_mean'][i],
                                df['nor_y_mean'][i],
                                df['nor_z_mean'][i]])
                if df['rake'][i] < 0:
                    rake = df['rake'][i]
                else:
                    rake = df['rake'][i]
                u, v, w = utilities_plot.slipvector_3D(p, r, nor, rake)
                xx = [x + (x - u), u]
                yy = [y + (y - v), v]
                zz = [z + (z - w), w]
                trace = go.Scatter3d(x=xx, y=yy, z=zz,
                                      mode='lines',
                                      line=dict(
                                          color=colors[i],
                                          width=6),
                                      legendgroup='Kinematics',
                                      name='Kinematics',
                                      hoverinfo='none',
                                      showlegend=legend_show[i],
                                      visible='legendonly'
                                      )
                fig.add_trace(trace)
        
        df_k = df[df['mean_dip'].isna()]
        df_k = df_k.reset_index(drop=True)
        trace = go.Scatter3d(
            x=df_k['_X'],
            y=df_k['_Y'],
            z=df_k['_Z'],
            mode='markers',
            marker=dict(
                color='rgb(0, 0, 0)',
                size=2,
                showscale=False),
            legendgroup='Kinematics',
            name='Kinematics',
            hoverinfo='none',
            showlegend=False,
            visible='legendonly',
            )
        fig.add_traces(trace)
        
    else:
        pass

    ############################################################################
    # Layout settings
    # Calculate the "empty white box" to ensure equal axes of the 3D plot
    # Work-around to ensure equal axes of the 3D plot
    x_range, y_range, z_range = utilities_plot.equal_axes(df['_X'],
                                            df['_Y'],
                                            df['_Z'])

    # Cameraview standard (Top view)
    eye = dict(x=0, y=-0.1, z=2)    
    
    # Define the figure layout parameters
    fig.update_layout(
        template='plotly_white',
        title=f'Hypocenter-Based Imaging of Active Faults (Truttmann et al. 2023): {project_title}',
        hovermode=None,
        showlegend=True,
        legend={'itemclick': 'toggle'},
        scene=dict(
            xaxis_title='Easting [m]',
            yaxis_title='Northing [m]',
            zaxis_title='Depth [m]',
            xaxis=dict(
                range=x_range,
                tickformat='d',
                separatethousands=True,
                showspikes=False,
                showgrid=True,
                zeroline=True
                ),
            yaxis=dict(
                range=y_range,
                tickformat='d',
                separatethousands=True,
                showspikes=False,
                showgrid=True,
                zeroline=True
                ),
            zaxis=dict(
                range=z_range,
                tickformat='d',
                separatethousands=True,
                showspikes=False,
                showgrid=True,
                zeroline=True
                ),
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=1),
            camera=dict(
                eye=eye)
            ),
        margin=dict(
            l=0,
            r=20,
            b=20,
            t=40)
        )
    
    fig.update_xaxes(title_standoff=20)
    
    # Save output
    out_path = os.path.join(input_params['out_dir'], 'Model_output')
    os.makedirs(out_path, exist_ok=True)

    fig.write_html(out_path + '/3D_model.html')

    return


def faults_stereoplot(input_params, data_output):
    # Unpack input parameters from dictionary
    for key, value in input_params.items():
        globals()[key] = value

    if 'class' in data_output.columns:
        column = data_output['class'].to_numpy()
        cmap = 'gnuplot'
        minval = np.nanmin(column) - 0.1
        maxval = np.nanmax(column) + 0.1
        colorsteps = 50
        colors = utilities_plot.colorscale_mplstereonet(column, cmap, minval, maxval, colorsteps, cmap_reverse=False)

    else:
        colors = ['black'] * len(data_output)

    column = data_output['kappa'].to_numpy()
    minval = 0
    maxval = np.nanmax(column)
    opacity = utilities_plot.opacity(column, minval, maxval, 20)
        
    fig, ax = mplstereonet.subplots()
    
    for i in range(len(data_output)):
        ax.pole((data_output['mean_azi'][i] - 90 % 360),
                data_output['mean_dip'][i],
                marker='o',
                c=colors[i],
                markersize=5,
                alpha=opacity[i])

    ax.set_azimuth_ticks(angles=[0, 180], labels=['North', 'South'])
    fig.set_figheight(6)
    fig.set_figwidth(6)
    
    # Save figure
    out_path = os.path.join(input_params['out_dir'], 'Model_output')
    os.makedirs(out_path, exist_ok=True)

    fig.savefig(out_path + '/Stereoplot.pdf')
    plt.close(fig)
    
    return
    

def nmc_histogram(input_params, data_input, per_X, per_Y, per_Z):
    """
    Parameters
    ----------
    per_X : TYPE
        DESCRIPTION.
    per_Y : TYPE
        DESCRIPTION.
    per_Z : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """   
    
    print('Plotting MC dataset histograms')
 
    mm = 1/25.4
    # Create output path (if not existing yet)
    out_path = os.path.join(input_params['out_dir'], 'Model_output')
    os.makedirs(out_path, exist_ok=True)
    newpath = out_path + '/ErrorDistributions'
    os.makedirs(newpath, exist_ok=True)
    
    # Precompute values outside the loop
    binwidth = 1
    linewidth1 = 1
    linewidth2 = 0.7
    plt.rcParams.update({'font.size': 8})


    for i in range(len(data_input)):
        fig, axs = plt.subplots(nrows=1, ncols=3,
                                figsize=(190*mm, 70*mm,), sharey=True,
                                tight_layout=True)

        axs[0].hist(per_X.loc[i, 1:], density=True,
                    bins=np.arange(min(per_X.loc[i, 1:]),
                               max(per_X.loc[i, 1:]) + binwidth, binwidth),
                    color='grey')
        axs[1].hist(per_Y.loc[i, 1:], density=True,
                    bins=np.arange(min(per_Y.loc[i, 1:]),
                               max(per_Y.loc[i, 1:]) + binwidth, binwidth),
                    color='grey')
        axs[2].hist(per_Z.loc[i, 1:], density=True,
                    bins=np.arange(min(per_Z.loc[i, 1:]),
                               max(per_Z.loc[i, 1:]) + binwidth, binwidth),
                    color='grey')
        
        y_axis_max = axs[0].get_ylim()[1]
        axs[0].vlines(data_input.loc[i, '_X'], 0, y_axis_max, color='black', linewidth=linewidth1)
        axs[1].vlines(data_input.loc[i, '_Y'], 0, y_axis_max, color='black', linewidth=linewidth1)
        axs[2].vlines(data_input.loc[i, '_Z'], 0, y_axis_max, color='black', linewidth=linewidth1)
        
        axs[0].vlines(data_input.loc[i, '_X'] + 0.5 * data_input.loc[i, 'EX'], 0, y_axis_max, color='black', linestyles='dashed', linewidth=linewidth2)
        axs[1].vlines(data_input.loc[i, '_Y'] + 0.5 * data_input.loc[i, 'EY'], 0, y_axis_max, color='black', linestyles='dashed', linewidth=linewidth2)
        axs[2].vlines(data_input.loc[i, '_Z'] + 0.5 * data_input.loc[i, 'EZ'], 0, y_axis_max, color='black', linestyles='dashed', linewidth=linewidth2)
        axs[0].vlines(data_input.loc[i, '_X'] - 0.5 * data_input.loc[i, 'EX'], 0, y_axis_max, color='black', linestyles='dashed', linewidth=linewidth2)
        axs[1].vlines(data_input.loc[i, '_Y'] - 0.5 * data_input.loc[i, 'EY'], 0, y_axis_max, color='black', linestyles='dashed', linewidth=linewidth2)
        axs[2].vlines(data_input.loc[i, '_Z'] - 0.5 * data_input.loc[i, 'EZ'], 0, y_axis_max, color='black', linestyles='dashed', linewidth=linewidth2)
        
        axs[0].set_xlabel('Easting X [m]')
        axs[1].set_xlabel('Northing Y [m]')
        axs[2].set_xlabel('Depth Z [m]')
        
        axs[0].set_ylabel('Probability density')
        
        # Set limits of histogram to ensure same scale of x-axis
        X_diff = abs(max(per_X.loc[i, 1:]) - min(per_X.loc[i, 1:]))
        Y_diff = abs(max(per_Y.loc[i, 1:]) - min(per_Y.loc[i, 1:]))
        Z_diff = abs(max(per_Z.loc[i, 1:]) - min(per_Z.loc[i, 1:]))
        max_range = 0.5 * max(X_diff, Y_diff, Z_diff)
        X_mean = data_input.loc[i, '_X']
        Y_mean = data_input.loc[i, '_Y']
        Z_mean = data_input.loc[i, '_Z']
        axs[0].set_xlim(X_mean - max_range, X_mean + max_range)
        axs[1].set_xlim(Y_mean - max_range, Y_mean + max_range)
        axs[2].set_xlim(Z_mean - max_range, Z_mean + max_range)
                
        ID = data_input.loc[i, 'ID']
                
        fig.savefig(newpath + f'/ErrorDist_{ID}.pdf')
        plt.close(fig)
        
    return


