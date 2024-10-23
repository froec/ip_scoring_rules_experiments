call conda activate base && ^
call python create_plots.py --plotdir results_acs/ && ^
call python create_plots.py --plotdir results_framingham/ && ^
call python create_plots.py --plotdir results_celeba/ && ^
call python create_entropy_comparison_plot.py