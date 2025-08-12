import matplotlib
matplotlib.use("TkAgg")  # For interactive hover in PyCharm

import matplotlib.pyplot as plt
import pandas as pd
import re
from matplotlib.lines import Line2D
from IPython import embed
import periodictable
import numpy as np


def multilinear_regression(database, y_param, x_params, labels):
    #for now ignore errobars for tau_imp 
    
    y = database[y_param]
    x = [database[x_param] for x_param in x_params]
    
    
    ln_y = np.log(y)
    ln_x = np.log(x)
        
    #add constant 
    X = np.vstack([np.ones_like(ln_y), ln_x]).T
    valid = np.isfinite(ln_y) & np.all(np.isfinite(X),axis=1)
    coeffs,s,ss,r = np.linalg.lstsq(X[valid], ln_y[valid], rcond=None)
    
    chi2 = s / valid.sum()
    model = np.dot(X[valid],coeffs)
    
    cov = np.linalg.inv(np.dot(X[valid].T, X[valid]))
    err = np.sqrt(np.diag(cov) * chi2)
    
    x_param = np.hstack(('const', x_params))
    labels = ['const.'] + labels
    significance = np.abs(coeffs)/err
    ind = np.argsort(-significance)
    
    print('Variable significance')
    
    print('param \t val \t err \t significance')
    for i in ind:
        print(f'{x_param[i]}\t {coeffs[i]:.2f}, {err[i]:.2f}, {significance[i]:.1f}' )
    
    plt.figure()
    std = X[valid].std(0)
    plt.bar(np.array(labels)[ind], -coeffs[ind]*std[ind], 
            yerr= err[ind]*std[ind], capsize=5)
    #plt.show()
    
    model_all = np.zeros_like(y) + np.nan
    model_all[valid] = model
    
    
    return np.exp(model_all)
    

 
def mds_or_ptdata_load(MDSconn, signal, shot):
    #it is not planned to be used regularly, it is wasting resources
    try:
        tdi = MDSconn.get('findsig("'+signal+'",_fstree)').value
        fstree = MDSconn.get('_fstree').value 
        MDSconn.openTree(fstree,shot)
        sig = MDSconn.get('_x='+tdi).value 
        
    except:
        sig = MDSconn.get(f'_x=PTDATA2("{signal}", {shot})').value 
    tvec = MDSconn.get('dim_of(_x)').value 
    #embed()
    return tvec, sig


def plot_timetraces(name):
    import MDSplus
    MDSconn  = MDSplus.Connection('localhost' )

    
    gid = '1277722485'
    id = '1fcbInMt-p0gMvjWdj3q6ZHCLHCX9OmnPWMQlbZFnvh'
    # "Impurity Confinement Time Database - Final Dataset (Shot).tsv"
    url = f"https://docs.google.com/spreadsheets/d/{id}/export?format=tsv&id={id}&gid={gid}"

    # Load and clean data
    df = pd.read_csv(url, sep="\t", header=[0,1,2])
    df.columns = df.columns.get_level_values(0)
 
    
    shots = np.unique(df.shot)
    f,ax = plt.subplots(1,1)

    for shot in shots:
        
        tvec, sig = mds_or_ptdata_load(MDSconn,  name, shot)
        
        subdf = df[df.shot==shot] 
        
        for tmin, tmax in zip(subdf.t_start, subdf.t_end):
            ax.axvspan(tmin, tmax, color='orange', alpha=0.3)
        ax.plot(tvec, sig)
        ax.set_ylim(sig.min(), sig.max())
        ax.set_xlim(tvec.min(), tvec.max())

        ax.set_title(shot)
        f.savefig(f'plots/{name}_{shot}.png')
        # f.clf()
        plt.cla()
    exit()

def plot_scatter(
        x_var,
        y_var,
        cluster=False,
        logx=False,
        logy=False,
        group_unknown=False,
        single_scenario=None,
        isotope='D',
        show_err=False, 
        xlim=None,
        ylim=None,
        impurity_group="all",
        ech_clusters=True,
        refline=False,
        ax = None,
        regression_vars = None,
):
    """
    Make a scatter plot from the TSV database.

    Parameters
    ----------
    x_var, y_var : str
    cluster : bool
    log : bool
    group_unknown : bool
    single_scenario : str | list[str] | None
    show_err: bool
    xlim, ylim : (float, float) | None
    impurity_group : {"all","high","low"}
    refline : bool
        If True, draw y = x reference line within the visible (overlapping) axis range.
    """
    
    # "Impurity Confinement Time Database - Final Dataset (Shot).tsv"
    url = "https://docs.google.com/spreadsheets/d/1fcbInMt-p0gMvjWdj3q6ZHCLHCX9OmnPWMQlbZFnvh8/export?format=tsv&id=1fcbInMt-p0gMvjWdj3q6ZHCLHCX9OmnPWMQlbZFnvh8&gid=1277722485"
    
    # gid = '1277722485'
    # id = '1fcbInMt-p0gMvjWdj3q6ZHCLHCX9OmnPWMQlbZFnvh'
    # # "Impurity Confinement Time Database - Final Dataset (Shot).tsv"
    # url = f"https://docs.google.com/spreadsheets/d/{id}/export?format=tsv&id={id}&gid={gid}"

    # Load and clean data
    df = pd.read_csv(url, sep="\t", header=[0,1,2])
    
    #extract header information
    col_names = df.columns.get_level_values(0)
    units = list(df.columns.get_level_values(1))
    latex = list(df.columns.get_level_values(2))
    

    #remove MultiIndex and replace with col names
    df.columns = col_names
    col_names = list(col_names)


        
    

    df["scenario"] = df["scenario"].fillna("").replace("", "Unknown")
    # Impurity classification
    LOW_Z = {"F", "Si", "Ca", "Cl", "Al"}
    HIGH_Z = {"Mo", "W", "Ni"}

        
    def impurity_atomic_number(val):
        if val == 'W':
            return 45 # typical W change in the plasma core 
        return getattr(periodictable, val).number
    
    #add impurity Z for plotting 
    df["impurity_Z"] = df["impurity"].apply(impurity_atomic_number)
    col_names.append('impurity_Z')
    units.append('-')
    latex.append('$Z$')
    
    if regression_vars is not None:
        labels = [latex[col_names.index(v)] for v in regression_vars]
        df["regression"] = multilinear_regression(df, y_var,  regression_vars,labels)
        
        col_names.append('regression')
        units.append(units[col_names.index(y_var)])
        latex.append('Regression model')
        x_var = 'regression'
        
    
    

    def classify_impurity(Z):
        return "High-Z" if Z > 20 else "Low-Z"

    df["impurity_group"] = df["impurity_Z"].apply(classify_impurity)

    key = str(impurity_group).strip()

    if key == "high":
        df = df[df["impurity_group"] == "High-Z"]
    elif key == "low":
        df = df[df["impurity_group"] == "Low-Z"]
    elif key == 'all':  # "all"
        df = df[df["impurity_group"].isin(["Low-Z", "High-Z"])]
    else:
        df = df[df["impurity"] == key]
        
    uscenario, counts = np.unique(df["scenario"], return_counts=True)
    
    
    
    def off_ech(rho):
        if np.isnan(rho):
            return 'noech'
        if rho > 0.4:
            return 'offaxis'
        return 'onaxis'
        
      
    df["ECH_loc"] = df["AOT:ECH_rho"].apply(off_ech)

    
    
    print('Statistics:')
    for us, c in zip(uscenario, counts):
        print(us, c)
    
    
    assert x_var in df.columns, f'x_var:{x_var} not in the database'
    assert y_var in df.columns, f'y_var:{y_var} not in the database'
    assert not show_err or y_var == 'tau_imp', 'errorbars availible only for tau_imp'
    
    
    
    # Scenario filtering
    allowed_scenarios = ["Hmode", "RMP Hmode", "WPQH", "QH mode", "Lmode", "NegD", 'Grassy hybrid', 'Hybrid']
    if group_unknown:
        allowed_scenarios.append("Unknown")
    other_scenarios = list(set(np.unique(df["scenario"])) - set(allowed_scenarios))
    
  
    df["scenario"] = df["scenario"].replace(other_scenarios, 'Others')
    
        
    #filter based on the plasma isotope
    df = df[df["isotope"] == isotope]
    
   
    if single_scenario is not None:
        if isinstance(single_scenario, str):
            selected_scenarios = [single_scenario]
        elif isinstance(single_scenario, list):
            selected_scenarios = single_scenario
        else:
            print("single_scenario must be None, a string, or a list of strings.")
            return
        df = df[df["scenario"].isin(selected_scenarios)].copy()
        if df.empty:
            print(f"No data found for scenario(s): {selected_scenarios}")
            return
        
    def rename_scenarios(val):
        scenario_names = {"Hmode": 'H-mode', "RMP Hmode": 'RMP H-mode',
         "WPQH": 'WPQH-mode', "QH mode": 'QH-mode', "Lmode": 'L-mode'}
        return scenario_names.get(val,val)

    df["scenario"] = df["scenario"].apply(rename_scenarios)



    # Ensure numeric axes columns
    df[x_var] = pd.to_numeric(df[x_var], errors="coerce")
    df[y_var] = pd.to_numeric(df[y_var], errors="coerce")
    
    df = df.dropna(subset=[x_var, y_var])
   
    if df.empty:
        print("No valid numeric data to plot after cleaning.")
        return

    # Labels (cluster or scenario)
    if cluster:
        def get_cluster(s):
            if group_unknown and s == "Unknown":
                return s
            return "Hmode-like" if s in ["H-mode", "RMP H-mode", "WPQH-mode", "QH-mode", 'Grassy Hybrid', 'Hybrid'] else "Lmode-like"

        df["label"] = df["scenario"].apply(get_cluster)

        style_map = {
            "Hmode-like": ("tab:blue", 'o'),
            "Lmode-like": ("tab:green", '^'),
            "Others": ('gray', 'd')
        }
    else:
        df["label"] = df["scenario"]
        style_map = {
            "H-mode": ('tab:blue', 'o'),
            "RMP H-mode": ('tab:purple', '*'),
            "WPQH-mode":( "tab:orange", 'X'),
            "QH-mode": ("tab:red", 'P'),
            "L-mode": ("tab:green", '<'),
            "NegD": ("tab:pink", '>'),
            'Grassy hybrid': ("k", 's'),
            'Hybrid': ("tab:brown", 'D'),
            "Others": ('gray', 'd'), 
        }


    #get latex label and units 
    xunits = units[col_names.index(x_var)]
    yunits = units[col_names.index(y_var)]
    xlatex = latex[col_names.index(x_var)]
    ylatex = latex[col_names.index(y_var)]
    
    
    # Plot
    if ax is None:
        fig_title = f"{y_var} vs {x_var}" + (f' (single_scenario)' if  single_scenario else '')
        fig, ax = plt.subplots(figsize=(5, 4), num=fig_title, dpi = 150)
    else:
        fig = gcf()
        
        
        
    scatter_meta = []
    from matplotlib.markers import MarkerStyle

    for label in df["label"].unique():
        sub_all = df[df["label"] == label]
        c, s = style_map.get(label, ("black", '+'))

        first_label = label
        
        if ech_clusters:
            iterator = [
                ("ECH_loc", "onaxis",  MarkerStyle(s, fillstyle='full')),
                ("ECH_loc", "offaxis", MarkerStyle(s, fillstyle='bottom')),
                ("ECH_loc", "noech",   MarkerStyle(s, fillstyle='none'))
            ]
        else:
            iterator = [
                ("impurity_group", "Low-Z",  MarkerStyle(s, fillstyle='full')),
                ("impurity_group", "High-Z", MarkerStyle(s, fillstyle='none'))
            ]
        
        for group, subgroup, ms in iterator:
            sub_low = sub_all[sub_all[group] == subgroup]
            if not sub_low.empty:
                line, = ax.plot(
                    sub_low[x_var], sub_low[y_var],
                    marker=ms, linestyle='',
                    markerfacecolor=c, markeredgecolor=c,
                    markersize=8,  # match s=40 from scatter
                    alpha=0.8, label=first_label
                )
                
                if show_err and y_var == 'tau_imp':
                    ax.errorbar(
                        sub_low[x_var], sub_low[y_var],
                        sub_low[y_var + '_error'], 
                        capsize=0, fmt=' ', ecolor=c
                    )
                
                vars = [x_var, y_var, "shot", "t_start", "t_end", "label", 'impurity']
                scatter_meta.append((line, sub_low[vars].to_numpy()))
                first_label = None

# Add text in lower right corner
    valid = np.isfinite(np.log(df[x_var])) & np.isfinite(np.log(df[y_var]))
    r = np.corrcoef(np.log(df[x_var])[valid], np.log(df[y_var])[valid])[0, 1]


    ax.text(0.95, 0.05, f'r = {r:.2f}', transform=ax.transAxes,
            ha='right', va='bottom', fontsize=10)


    # Axis settings
    
    #detect of log scale is useful
    if logx is None:
        logx = np.all(df[x_var] > 0) and df[x_var].max()/df[x_var].min() > 5 
    if logy is None:
        logy = np.all(df[y_var] > 0) and df[y_var].max()/df[y_var].min() > 5 
         
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")

    def _normalize_limits(lim, name, is_log):
        if lim is None:
            return None
        if not isinstance(lim, (tuple, list)) or len(lim) != 2:
            print(f"{name} must be a (min, max) tuple/list; ignoring.")
            return None
        lo, hi = lim
        if pd.isna(lo) or pd.isna(hi):
            print(f"{name} has NaN; ignoring.")
            return None
        if lo > hi:
            lo, hi = hi, lo
        if is_log and lo <= 0:
            print(f"{name} must be > 0 for log scale; ignoring.")
            return None
        return (lo, hi)

    xlim_use = _normalize_limits(xlim, "xlim", logx)
    ylim_use = _normalize_limits(ylim, "ylim", logy)
    if xlim_use is not None:
        ax.set_xlim(xlim_use)
    if ylim_use is not None:
        ax.set_ylim(ylim_use)

    # Optional reference line y = x (uses overlapping visible range)
    if refline:
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        lo = max(min(x0, x1), min(y0, y1))
        hi = min(max(x0, x1), max(y0, y1))
        #one positive point for log scaled axis
        pos_lo = min(df[x_var][df[x_var]>0].min(),df[y_var][df[y_var]>0].min())

        ax.plot([lo,pos_lo, hi], [lo,pos_lo, hi], linestyle="--", color="k", linewidth=1, alpha=0.7)

    # Labels and title
    if ech_clusters:
        imp_txt = ''
    else:
        imp_label = {"all": "Lower Z & Higher Z", "low": "Lower Z only", "high": "Higher Z only"}
        imp_txt = imp_label.get(key, key)
        
    base_title = f"{ylatex} vs {xlatex}"# + (f' ({single_scenario})' if  single_scenario else '')

    ax.set_title(f"{base_title} {imp_txt}", fontsize=14)
    ax.set_xlabel(f"{xlatex} [{xunits}]", fontsize=12)
    ax.set_ylabel(f"{ylatex} [{yunits}]", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.6)

    # Legends: scenario/cluster (top-right) and impurity (top-left)
    leg1 = ax.legend(
        title=("Scenario" if not cluster else "Cluster"),
        fontsize=9,
        loc="upper right",
        bbox_to_anchor=(1.0, 1.0),
        draggable=True
    )
    ax.add_artist(leg1)
    
    if ech_clusters:
              
        common = dict(linestyle='', marker='o', markersize=8,
                    markerfacecolor='black', markeredgecolor='black')

        marker_proxies = [
            ax.plot([], [], fillstyle='full',   label="On-axis",  **common)[0],
            ax.plot([], [], fillstyle='bottom', label="Off-axis", **common)[0],
            ax.plot([], [], fillstyle='none',   label="No ECH",   **common)[0],
        ]
        
  
        leg2 = ax.legend(
            handles=marker_proxies,
            title="ECH location",
            fontsize=9,
            loc="upper left",
            bbox_to_anchor=(0.0, 1.0),
            draggable=True
        )
        ax.add_artist(leg2)        
    else:
        if key == 'all':
            marker_proxies = [
                    Line2D([0], [0], marker="o", linestyle="None", markersize=6,
                        markerfacecolor="black", markeredgecolor="k", label="Lower Z (filled)"),
                    Line2D([0], [0], marker="o", linestyle="None", markersize=6,
                        markerfacecolor="none", markeredgecolor="black", label="Higher Z (unfilled)"),
                ]
            leg2 = ax.legend(
                handles=marker_proxies,
                title="Impurity",
                fontsize=9,
                loc="upper left",
                bbox_to_anchor=(0.0, 1.0),
                draggable=True
            )
            ax.add_artist(leg2)

    plt.tight_layout()

    annot = ax.annotate("", xy=(0, 0), xytext=(10, 10),
                        textcoords="offset points",
                        bbox=dict(boxstyle="round", fc="w"),
                        arrowprops=dict(arrowstyle="->"))
    annot.set_visible(False)

    def update_annot(event, x, y, shot, t_start, t_end, label, imp):
        annot.xy = (x, y)
        text = (f"Shot: {int(shot)}\n"
                f"Time: {int(t_start)} - {int(t_end)} ms\n"
                f"{label}\nimp: {imp}\n"
                f"{xlatex} = {x:.3g}\n{ylatex} = {y:.3g}")
        annot.set_text(text)
        annot.get_bbox_patch().set_alpha(0.9)
        if event:
            fig_width, fig_height = fig.get_size_inches() * fig.dpi
            y_offset = -80 if event.y > fig_height / 2 else 40
            x_offset = -140 if event.x > fig_width / 2 else 10
            annot.set_position((x_offset, y_offset))

    def hover(event):
        vis = annot.get_visible()
        if event.inaxes == ax:
            for artist, meta in scatter_meta:
                cont, ind = artist.contains(event)
         
                # For both PathCollection (scatter) and Line2D (plot)
                if  cont and "ind" in ind and len(ind["ind"]):
             
                    i = ind["ind"][0]
                    update_annot(event, *meta[i])
                    annot.set_visible(True)
                    fig.canvas.draw_idle()
                    return
        if vis:
            annot.set_visible(False)
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", hover)
    plt.show()

if __name__ == "__main__":
    # group Hmode-like / Lmode-like
    cluster_flag = False

    # log scale axes True, False, None is auto
    log_flag = None

    # include unknown scenarios
    unknown_flag = True

    # str for single scenario, list(str) for multiple, None for all
    single_scenario_flag = None
    single_scenario_flag = ["Hmode", "RMP Hmode", 'Grassy hybrid', 'Hybrid']#,  "WPQH", "QH mode",]

    # tuple limits for x axis, None for autoscale
    x_limits = None

    # tuple limits for y axis, None for autoscale
    y_limits = None

    # "all", "low" (Ca and lighter), or "high" (heavier than Ca) Z impurities
    #or impurity name
    impurity_subset = "all"

    # True to draw y = x
    draw_refline = False
    
    
    y_variable = "AOT_tauimp_tauth"
    x_variable = "IDA_Te_Ti"
    
    y_variable = "AOT_tauimp_tauth"
    x_variable = "IDA_nu_star"
    #y_variable = "IDA_Te_Ti"

 

    
    regression_vars = None

    #regression_vars = ['IDA_beta_tor_e', 'IDA_Te_Ti', 'IDA_nu_star', 'IDA_mach', 'impurity_Z']
    
    #regression_vars = ['mach', 'Te/Ti', 'EFIT01_q95', 'EFIT01_betan', 'impurity_Z', 'EFIT01_kappa', 'EFIT01_tritop']


    #regression_vars = ['mach', 'Te/Ti', 'EFIT01_q95', 'EFIT01_betan', 'impurity_Z', 'EFIT01_kappa', 'EFIT01_tritop', 'Pe/Pi']


    #regression_vars = ['triangularity',  'PTDATA_bt', 'EFIT01_density', 'EFIT01_kappa', 'Pe/Pi']


    #y_variable = "tau_imp"



    #x_variable = y_variable
 
 

    plot_scatter(
        x_variable,
        y_variable,
        cluster=cluster_flag,
        logx=log_flag,
        logy=log_flag,
        group_unknown=unknown_flag,
        single_scenario=single_scenario_flag,
        xlim=x_limits,
        ylim=y_limits,
        show_err=False,
        isotope='D',
        impurity_group=impurity_subset,
        ech_clusters = True,
        refline=draw_refline,
        regression_vars = regression_vars
    )


# 
# ([
#        'isotope', 'impurity', 'injection time', 'script_used', 'channel',
#        'tau_imp', 'tau_imp_error', 'method', 'Planned scenario',
#        'shot comments', 't_start', 't_end', 'time_range', 'filterscope',
#        '~ELM_count', '~ELM_freq', 'PTDATA_ip', 'PTDATA_bt', 'max(i_coil)',
#        'PTDATA_il30', 'PTDATA_iu30', 'PTDATA_il90', 'PTDATA_iu90',
#        'PTDATA_il150', 'PTDATA_iu150', 'PTDATA_il210', 'PTDATA_iu210',
#        'PTDATA_il270', 'PTDATA_iu270', 'PTDATA_il330', 'PTDATA_iu330',
#        'EFIT01_density', 'EFIT01_betan', 'EFIT01_betap', 'EFIT01_q95',
#        'EFIT01_r0', 'EFIT01_aminor', 'EFIT01_tritop', 'EFIT01_tribot',
#        'EFIT01_kappa', 'EFIT01_area', 'TRANSPORT_taue', 'TRANSPORT_tauth',
#        'TRANSPORT_ptot', 'TRANSPORT_h_thh98y2', 'TRANSPORT_tauthh98y2',
#        'RF_echpwrc', 'ELECTRONS_TSTE_TAN', 'ELECTRONS_TSNE_TAN',
#        'ELECTRONS_prmtan_neped', 'ELECTRONS_prmtan_peped',
#        'ELECTRONS_prmtan_teped', 'IONS_cerqtit1', 'IONS_cerqtit2',
#        'IONS_cerqtit3', 'IONS_cerqtit17', 'IONS_cerqtit18', 'IONS_cerqtit19',
#        'IONS_cerqrotct1', 'IONS_cerqrotct2', 'IONS_cerqrotct3',
#        'IONS_cerqrotct17', 'IONS_cerqrotct18', 'IONS_cerqrotct19',
#        'IONS_ceratit1', 'IONS_ceratit2', 'IONS_ceratit3', 'IONS_ceratit17',
#        'IONS_ceratit18', 'IONS_ceratit19', 'IONS_cerarotct1',
#        'IONS_cerarotct2', 'IONS_cerarotct3', 'IONS_cerarotct17',
#        'IONS_cerarotct18', 'IONS_cerarotct19', 'MHD_n1rms', 'MHD_n2rms',
#        'SPECTROSCOPY_zeff', 'nu_star', 'mean_Te', 'mean_Ti', 'mean_vtor',
#        'mach', 'Te/Ti', 'ITERH-98P(y,2)', 'H98 (tauth)', 'H98 (taue)',
#        'triangularity', 'derived_kappa'],
