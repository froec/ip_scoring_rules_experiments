import logging
import argparse
import numpy as np
import pandas as pd
import re
from utils import SimpleLogger

import matplotlib.pyplot as plt
import seaborn as sns


# Example usage: python create_plots.py --plotdir results_framingham/

logger = SimpleLogger('logger', log_file='evaluation.log', level=logging.DEBUG)
def parse_args():
    parser = argparse.ArgumentParser(description="Parse arguments for the model.")
    parser.add_argument("--plotdir", type=str, default="models/", help="Directory for reading evaluation results and saving plots.")
    parser.add_argument("--show_plots", type=int, default=0, help="Show the created plots. 0=don't show, 1=show")
    return parser.parse_args()

args = parse_args()
plotdir = args.plotdir

########################################
######## Load the outputs of the evaluation.py script
ipscores_df = pd.read_csv(plotdir + 'ipscores_df.csv')
all_combined_res_df = pd.read_csv(plotdir + 'all_combined_res_df.csv')
ip_dec_cal_df = pd.read_csv(plotdir + 'ip_dec_cal_df.csv')


"""
# filter the results
def filter_df(df):
    lossname_set = ["CostSensitive(c=0.1)","CostSensitive(c=0.3)","CostSensitive(c=0.7)","CostSensitive(c=0.9)"]
    models_set = ["model-log","model-log-dro","model-asymm(c=0.1)-dro","model-asymm(c=0.3)-dro","model-asymm(c=0.7)-dro","model-asymm(c=0.9)-dro","GBR"]
    df = df[df['Predictor Name'].isin(models_set)]
    df = df[df['Loss Name'].isin(lossname_set)]
    return df 

ipscores_df = filter_df(ipscores_df)
all_combined_res_df = filter_df(all_combined_res_df)
ip_dec_cal_df = filter_df(ip_dec_cal_df)
"""


########################################
######## visualization of IP scores
# with respect to IP data model

sequential_cmap = plt.cm.Oranges
loss_funcs = list(set(ipscores_df["Loss Name"]))
n_colors = len(loss_funcs)
# Create a color palette
colors = sequential_cmap(np.linspace(0.2, 0.8, n_colors))

#ipscores_df = ipscores_df[ipscores_df["Loss Name"]=="CostSensitive(c=0.1)"]

# first train data model
# here each distribution in the envelope represents the training distribution for one state
# and they are combined to obtain the train IP data model
plt.figure(figsize=(12, 6))
sns.barplot(x='Predictor Name', y='IP Score (Train)', hue='Loss Name', data=ipscores_df, palette=colors)

# Customize the plot
plt.title('IP Scores (wrt train IP data model) for different predictors across loss functions (IP scoring rules)')
plt.xlabel('Predictor')
plt.ylabel('IP Score')


# Rotate x-axis labels if they overlap
plt.xticks(rotation=45, ha='right')

# Adjust layout and show legend
plt.tight_layout()
plt.legend(title='Loss', loc='upper left')

plt.savefig(args.plotdir + "ip_scores_train.pdf", bbox_inches='tight')
if args.show_plots>=1:
    plt.show()
else:
    plt.close()


# now test data model
plt.figure(figsize=(12, 6))
sns.barplot(x='Predictor Name', y='IP Score (Test)', hue='Loss Name', data=ipscores_df, palette=colors)

# Customize the plot
plt.title('IP Scores (wrt test IP data model) for different predictors across loss functions (IP scoring rules)')
plt.xlabel('Predictor')
plt.ylabel('IP Score')

# Rotate x-axis labels if they overlap
plt.xticks(rotation=45, ha='right')

# Adjust layout and show legend
plt.tight_layout()
plt.legend(title='Loss',  loc='upper left')

plt.savefig(args.plotdir + "ip_scores_test.pdf", bbox_inches='tight')
if args.show_plots>=1:
    plt.show()
else:
    plt.close()


######## helpers
def shortenLossNames(lossname):
        match = re.search(r"CostSensitive\(c=([0-9.]+)\)", lossname)
        if match:
            #return f"CS({match.group(1)})"
            return r"$\ell_{%s}$"%match.group(1)
        return lossname

def shortenPredictorNames(predictorname):
    match = re.search(r"model-asymm\(c=([0-9.]+)\)-dro", predictorname)
    if match:
        #return f"CS({match.group(1)})"
        return r"DRO($%s$)"%match.group(1)

    if predictorname == "model-log-dro":
        return "DRO(log)"

    if predictorname == "model-combined-log":
        return "ERM(log)"
    return predictorname



######## the above barplots as latex tables:
for train_or_test in ["Train","Test"]:

    pivot_df = ipscores_df.pivot(index='Predictor Name', columns='Loss Name', values=('IP Score (%s)'%train_or_test))
    pivot_df.columns = [shortenLossNames(c) for c in pivot_df.columns]
    pivot_df.index = pivot_df.index.map(shortenPredictorNames)
    pivot_df = pivot_df.rename_axis('Predictor', axis='index')




    def highlight_min_in_column(column):
        is_min = column == column.min()
        return ['\\textbf{' + ('%.4f'%val) + '}' if min_val else ('%.4f'%val) for val, min_val in zip(column, is_min)]

    # Apply the highlight function to all columns in the DataFrame
    bolded_df = pivot_df.apply(highlight_min_in_column, axis=0)

    # Export to LaTeX
    latex_table = bolded_df.to_latex(index=True, escape=False, )  # escape=False allows LaTeX formatting like \textbf{}

    f = open(args.plotdir+"ip_scores_%s_table.txt"%train_or_test, "w")
    f.write(latex_table)
    f.close()

    # Print the LaTeX table
    print(train_or_test)
    print("");
    print(latex_table)
    print("");print("");






########################################
######## visualization of train/test losses on combined (!) data 
# that is, with respect to precise data model
# first visualize train losses
plt.figure(figsize=(12, 6))
sns.barplot(x='Predictor Name', y='Train Loss', hue='Loss Name', data=all_combined_res_df, palette=colors)

# Customize the plot
plt.title('Train Losses (combined data) for different predictors across loss functions')
plt.xlabel('Predictor')
plt.ylabel('Train Loss')

# Rotate x-axis labels if they overlap
plt.xticks(rotation=45, ha='right')

# Adjust layout and show legend
plt.tight_layout()
plt.legend(title='Loss',  loc='upper left')

plt.savefig(args.plotdir + "losses_train_combined_data.pdf", bbox_inches='tight')
if args.show_plots>=1:
    plt.show()
else:
    plt.close()


# then the same for test losses
plt.figure(figsize=(12, 6))
sns.barplot(x='Predictor Name', y='Test Loss', hue='Loss Name', data=all_combined_res_df, palette=colors)

# Customize the plot
plt.title('Test Losses (combined data) for different predictors across loss functions')
plt.xlabel('Predictor')
plt.ylabel('Test Loss')

# Rotate x-axis labels if they overlap
plt.xticks(rotation=45, ha='right')

# Adjust layout and show legend
plt.tight_layout()
plt.legend(title='Loss', loc='upper left')

plt.savefig(args.plotdir + "losses_test_combined_data.pdf", bbox_inches='tight')
if args.show_plots>=1:
    plt.show()
else:
    plt.close()






def shortenDecCalTypeNames(deccaltype):
    if deccaltype == "IP DecCal (Train)":
        return "Overall IP Cal. (Train)"
    if deccaltype == "IP DecCal (Test)":
        return "Overall IP Cal. (Test)"
    if deccaltype == "DecCal (Train) : a=0":
        return "IP Cal. on \"a=0\" group (Train)"
    if deccaltype == "DecCal (Train) : a=1":
        return "IP Calibration on \"a=1\" group (Train)"
    if deccaltype == "DecCal (Test) : a=0":
        return "IP Cal. on \"a=0\" group (Test)"
    if deccaltype == "DecCal (Test) : a=1":
        return "IP Cal. on \"a=1\" group (Test)"
    else:
        sys.exit()


########################################
########### visualize IP decision calibration with action induced partitions 
for train_or_test in ['Train','Test']:
    melted_df = ip_dec_cal_df.melt(
        id_vars=['Predictor Name', 'Loss Name'],
        value_vars=['IP DecCal (%s)'%train_or_test, 'DecCal (%s) : a=0'%train_or_test, 'DecCal (%s) : a=1'%train_or_test],
        var_name='DecCal Type',
        value_name='DecCal Value'
    )
    melted_df["Predictor Name"] = melted_df["Predictor Name"].map(shortenPredictorNames)
    melted_df["Loss Name"] = melted_df["Loss Name"].map(shortenLossNames)

    melted_df = melted_df.rename(columns={'Loss Name': 'Loss'})

    sequential_cmap = plt.cm.Oranges
    n_colors = len(loss_funcs)
    # Create a color palette
    colors = sequential_cmap(np.linspace(0.2, 0.8, n_colors))

    g = sns.catplot(
        data=melted_df, 
        x='Predictor Name', 
        y='DecCal Value', 
        hue='Loss', 
        col='DecCal Type',
        kind='bar',
        height=6, 
        aspect=1.25,
        sharey=False,
        palette=colors
    )

    # Rotate x-axis labels for better readability
    g.set_xticklabels(rotation=45, ha='right')

    #g.legend.set(loc='lower left')

    # Adjust the layout to prevent overlapping
    plt.tight_layout()


    plt.savefig(args.plotdir + ("ip_dec_cal_%s.pdf" % train_or_test), bbox_inches='tight')

    # Show the plot
    if args.show_plots>=1:
        plt.show()
    else:
        plt.close()



    # now do this again but only for overall decision calibration and a=0
    # types are:
    # IP DecCal (Train)
    # DecCal (Train) : a=0
    # DecCal (Train) : a=1
    print(print(melted_df["DecCal Type"].unique()))
    df2 = melted_df[melted_df["DecCal Type"].isin(["IP DecCal (%s)"%train_or_test,"DecCal (%s) : a=0"%train_or_test])]
    df2["DecCal Type"] = df2["DecCal Type"].map(shortenDecCalTypeNames)
    
    fs = 1.1
    with sns.plotting_context('notebook',font_scale=fs): 
        g = sns.catplot(
            data=df2, 
            x='Predictor Name', 
            y='DecCal Value', 
            hue='Loss', 
            col='DecCal Type',
            kind='bar',
            height=4, 
            aspect=1.25,
            sharey=False,
            palette=colors
        )
        g.set_titles("{col_name}")

        # Rotate x-axis labels for better readability
        g.set_xticklabels(rotation=45, ha='right')

        g.set_ylabels("Diagnostic Value")
        g.set_xlabels("")

        g.legend.set(loc='upper right')

        # Adjust the layout to prevent overlapping
        plt.tight_layout()

        plt.savefig(args.plotdir + ("ip_dec_cal_%s_subset.pdf" % train_or_test), bbox_inches='tight')

        # Show the plot
        if args.show_plots>=1:
            plt.show()
        else:
            plt.close()



        # now the plot for only a=1
        df2 = melted_df[melted_df["DecCal Type"].isin(["DecCal (%s) : a=1"%train_or_test])]
        df2["DecCal Type"] = df2["DecCal Type"].map(shortenDecCalTypeNames)
        
        fs = 1.1
        with sns.plotting_context('notebook',font_scale=fs): 
            g = sns.catplot(
                data=df2, 
                x='Predictor Name', 
                y='DecCal Value', 
                hue='Loss', 
                col='DecCal Type',
                kind='bar',
                height=4, 
                aspect=1.25,
                sharey=False,
                palette=colors
            )
            g.set_titles("{col_name}")

            # Rotate x-axis labels for better readability
            g.set_xticklabels(rotation=45, ha='right')

            g.set_ylabels("Diagnostic Value")
            g.set_xlabels("")

            g.legend.set(loc='upper right')

            # Adjust the layout to prevent overlapping
            plt.tight_layout()

            plt.savefig(args.plotdir + ("ip_dec_cal_%s_subset2.pdf" % train_or_test), bbox_inches='tight')

            # Show the plot
            if args.show_plots>=1:
                plt.show()
            else:
                plt.close()


        # now the plot for a=0 and a=1
        df2 = melted_df[melted_df["DecCal Type"].isin(["DecCal (%s) : a=0"%train_or_test,"DecCal (%s) : a=1"%train_or_test])]
        df2["DecCal Type"] = df2["DecCal Type"].map(shortenDecCalTypeNames)
        
        fs = 1.1
        with sns.plotting_context('notebook',font_scale=fs): 
            g = sns.catplot(
                data=df2, 
                x='Predictor Name', 
                y='DecCal Value', 
                hue='Loss', 
                col='DecCal Type',
                kind='bar',
                height=4, 
                aspect=1.25,
                sharey=False,
                palette=colors
            )
            g.set_titles("{col_name}")

            # Rotate x-axis labels for better readability
            g.set_xticklabels(rotation=45, ha='right')

            g.set_ylabels("Diagnostic Value")
            g.set_xlabels("")

            g.legend.set(loc='upper right')

            # Adjust the layout to prevent overlapping
            plt.tight_layout()

            plt.savefig(args.plotdir + ("ip_dec_cal_%s_subset3.pdf" % train_or_test), bbox_inches='tight')

            # Show the plot
            if args.show_plots>=1:
                plt.show()
            else:
                plt.close()







########### new plot layout
df = ip_dec_cal_df.melt(
        id_vars=['Predictor Name', 'Loss Name'],
        value_vars=['IP DecCal (Train)','IP DecCal (Test)', 'DecCal (Train) : a=0', 'DecCal (Train) : a=1',\
                        'DecCal (Test) : a=0', 'DecCal (Test) : a=1'],
        var_name='DecCal Type',
        value_name='DecCal Value'
    )

print(df)
df["Predictor Name"] = df["Predictor Name"].map(shortenPredictorNames)
df["Loss Name"] = df["Loss Name"].map(shortenLossNames)

df = df.rename(columns={'Loss Name': 'Loss'})

sequential_cmap = plt.cm.Oranges
n_colors = len(loss_funcs)
# Create a color palette
colors = sequential_cmap(np.linspace(0.2, 0.8, n_colors))


filterlist = [  (1, ["IP DecCal (Train)", "IP DecCal (Test)"]),\
                (2, ["DecCal (Train) : a=0", "DecCal (Test) : a=0"]),\
                (3, ["DecCal (Train) : a=1", "DecCal (Test) : a=1"])]

for i,filterl in filterlist:
    df2 = df[df["DecCal Type"].isin(filterl)]
    df2["DecCal Type"] = df2["DecCal Type"].map(shortenDecCalTypeNames)

    fs = 1.1
    with sns.plotting_context('notebook',font_scale=fs): 
        g = sns.catplot(
            data=df2, 
            x='Predictor Name', 
            y='DecCal Value', 
            hue='Loss', 
            col='DecCal Type',
            kind='bar',
            height=4, 
            aspect=1.25,
            sharey=False,
            palette=colors
        )
        g.set_titles("{col_name}")

        # Rotate x-axis labels for better readability
        g.set_xticklabels(rotation=45, ha='right')

        g.set_ylabels("Diagnostic Value")
        g.set_xlabels("")

        g.legend.set(loc='upper right')

        from matplotlib.ticker import FuncFormatter
        def format_with_sign(x, p):
            return f"{x:+.2f}"
        def format_y_labels(g):
            for ax in g.axes.flat:
                ax.yaxis.set_major_formatter(FuncFormatter(format_with_sign))
        format_y_labels(g)

        # Adjust the layout to prevent overlapping
        plt.tight_layout()


        plt.savefig(args.plotdir + "ip_dec_cal_row%s.pdf"%i , bbox_inches='tight')

        # Show the plot
        if args.show_plots>=1:
            plt.show()
        else:
            plt.close()