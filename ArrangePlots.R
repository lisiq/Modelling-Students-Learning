rm(list = ls())
library(ggplot2)
#library(dplyr)
library(ggpubr)
library(png)

setwd(getSrcDirectory(function(){})[1])

FIG_DIR = './vis/'

DPI = 800
WIDTH = 6.4
HEIGHT = 4.8 # 1 row

FONT.LABEL = list(size = 20, color = "black", face = "bold", family = NULL)


read_fig = function(filename){
  print(filename)
  img = readPNG(file.path(FIG_DIR, filename))
  
  myplot = ggplot() + background_image(img) + theme_void()
  
  return(myplot)
}

read_list = function(fig_list){
  figs = lapply(fig_list, read_fig)
  
  #names(figs) = names(fig_list)
  return(figs)
}

save_fig = function(figname, fig_list, nrow, ncol, wlabels=T){
  if (wlabels) labels = names(fig_list) 
  else labels = NULL
  
  myfigs = ggarrange(plotlist = read_list(fig_list),
    labels = labels, ncol=ncol, nrow=nrow, font.label = FONT.LABEL)
  ggsave(filename = file.path(OUT_DIR, figname), 
    plot = myfigs,
    dpi = DPI,
    width = WIDTH*ncol,   
    height = HEIGHT*nrow,
    units = "in")
}



########################
# Paper Figures
########################

OUT_DIR = file.path(FIG_DIR,  'paper') 

# maybe change by scales
if (T) {
  
# 
#   item_fig_list = list(
#     '_' = 'SAGE_scales_full_items_PCA.png',
#     'A' = 'SAGE_scales_full_dim_items/type_reg-x-None-var_IRT1_difficulty.png',
#     'B' = 'SAGE_scales_full_dim_items/type_reg-y-None-var_IRT1_difficulty.png',
#     'C' = 'SAGE_scales_full_dim_items/type_reg-z-None-var_IRT1_difficulty.png',
#     'D' = 'SAGE_scales_full_dim_items/type_reg-u-None-var_IRT1_difficulty.png',
#     'E' = 'SAGE_scales_full_dim_items/type_reg-v-None-var_IRT1_difficulty.png',
#     'F' = 'SAGE_scales_full_dim_items/type_reg-x-None-var_IRT1_discrimination_transf.png', 
#     'G' = 'SAGE_scales_full_dim_items/type_reg-y-None-var_IRT1_discrimination_transf.png', 
#     'H' = 'SAGE_scales_full_dim_items/type_reg-z-None-var_IRT1_discrimination_transf.png',
#     'I' = 'SAGE_scales_full_dim_items/type_reg-u-None-var_IRT1_discrimination_transf.png', 
#     'J' = 'SAGE_scales_full_dim_items/type_reg-v-None-var_IRT1_discrimination_transf.png'
#   )

  item_fig_list = list(
    '_' = 'SAGE_scales_full_items_PCA.png',
    'A' = 'SAGE_scales_full_dim_items/type_reg-x-None-var_IRT1_difficulty.png',
    'B' = 'SAGE_scales_full_dim_items/type_reg-y-None-var_IRT1_difficulty.png',
    'C' = 'SAGE_scales_full_dim_items/type_reg-z-None-var_IRT1_difficulty.png',
    'D' = 'SAGE_scales_full_dim_items/type_reg-x-None-var_IRT1_discrimination_transf.png', 
    'E' = 'SAGE_scales_full_dim_items/type_reg-y-None-var_IRT1_discrimination_transf.png', 
    'F' = 'SAGE_scales_full_dim_items/type_reg-z-None-var_IRT1_discrimination_transf.png'
  )
  
  item_notenc_fig_list = list(
    'A' = 'SAGE_scales_full_dim_items_notenc/type_reg-x-None-var_IRT1_difficulty.png',
    'B' = 'SAGE_scales_full_dim_items_notenc/type_reg-y-None-var_IRT1_difficulty.png',
    'C' = 'SAGE_scales_full_dim_items_notenc/type_reg-z-None-var_IRT1_difficulty.png',
    'D' = 'SAGE_scales_full_dim_items_notenc/type_reg-u-None-var_IRT1_difficulty.png',
    'E' = 'SAGE_scales_full_dim_items_notenc/type_reg-v-None-var_IRT1_difficulty.png',
    'F' = 'SAGE_scales_full_dim_items_notenc/type_reg-x-None-var_IRT1_discrimination_transf.png', 
    'G' = 'SAGE_scales_full_dim_items_notenc/type_reg-y-None-var_IRT1_discrimination_transf.png', 
    'H' = 'SAGE_scales_full_dim_items_notenc/type_reg-z-None-var_IRT1_discrimination_transf.png',
    'I' = 'SAGE_scales_full_dim_items_notenc/type_reg-u-None-var_IRT1_discrimination_transf.png', 
    'J' = 'SAGE_scales_full_dim_items_notenc/type_reg-v-None-var_IRT1_discrimination_transf.png'
  )
  
  #    'G' = 'SAGE_scales_full_dim_items_notenc/type_reg-x-None-var_IRT1_difficulty.png',
  #'H' = 'SAGE_scales_full_dim_items_notenc/type_reg-y-None-var_IRT1_difficulty.png',
  #'I' = 'SAGE_scales_full_dim_items_notenc/type_reg-z-None-var_IRT1_difficulty.png',
  #'J' = 'SAGE_scales_full_dim_items_notenc/type_reg-x-None-var_IRT1_discrimination_transf.png', 
  #'K' = 'SAGE_scales_full_dim_items_notenc/type_reg-y-None-var_IRT1_discrimination_transf.png', 
  #'L' = 'SAGE_scales_full_dim_items_notenc/type_reg-z-None-var_IRT1_discrimination_transf.png'
  


  distances_fig_list = list(
    'A' = 'SAGE_scales_full_items/type_sct-x-y-var_scale.png',
    'B' = 'SAGE_scales_full_items/type_sct-z-y-var_scale.png',
    'C' = 'SAGE_scales_full_scale_distance_bw.png',
    'D' = 'SAGE_scales_full_items/type_sct-x-y-var_domain.png',
    'E' = 'SAGE_scales_full_items/type_sct-z-y-var_domain.png',
    'F' = 'SAGE_scales_full_domain_distance_bw.png'
    
  )
  
  # show the effect of competences
  cluster_fig_list = list(
    'A' = 'SAGE_scales_full_matrix_bw_random_CH.png',
    'B' = 'SAGE_scales_full_matrix_bw_random_DB.png',
    'C' = 'SAGE_scales_full_scalexdifficulty_matrix_clustering_CH.png',
    'D' = 'SAGE_scales_full_scalexdifficulty_matrix_clustering_DB.png'
  )
  
  # show the effect of topics
  cluster_supp_fig_list2 = list(
    'A' = 'SAGE_scales_full_scalexdifficulty_topic_clustering_CH.png',
    'B' = 'SAGE_scales_full_scalexdifficulty_topic_clustering_DB.png',
    'C' = 'SAGE_scales_topic_scalexdifficulty_topic_clustering_CH.png',
    'D' = 'SAGE_scales_topic_scalexdifficulty_topic_clustering_DB.png'
  )
  
  
  # show that the similarities are not due to being in the same session
  cluster_supp_fig_list1 = list(
    'A' = 'SAGE_scales_matrix_scalexdifficulty_matrix_clustering_CH.png',
    'B' = 'SAGE_scales_matrix_scalexdifficulty_matrix_clustering_DB.png'
  )
  
  student_fig_list = list(
    'A' = 'SAGE_scales_full_students_PCA.png',
    'B' = 'SAGE_scales_full_students/type_sct-x-y-var_Gender_motherTongue.png',
    'C' = 'SAGE_scales_full_students/type_sct-z-y-var_Gender_motherTongue.png',
    'D' = 'SAGE_scales_full_dim_students/type_kde-x-None-var_Gender_motherTongue.png',
    'E' = 'SAGE_scales_full_dim_students/type_kde-y-None-var_Gender_motherTongue.png',
    'F' = 'SAGE_scales_full_dim_students/type_kde-z-None-var_Gender_motherTongue.png'
  )
  
  #edge_fig_list = list(
    #'A' = 'SAGE_full_edges_PCA.png',
  #  'A' = 'SAGE_full_age_edges_agg_wl.png',
  #  'B' = 'SAGE_full_ability_edges_agg_wl.png',
  #  'C' = 'SAGE_full_previous_sessions_edges.png',
  #  'D' = 'SAGE_full_frequency_edges.png'
  #)
  # frequency/ amount
  # sotphere
  # 
  # myfigs2 = ggarrange(plotlist = read_list(item_fig_list[2:7]),
  #   labels = labels, ncol=3, nrow=2, font.label = FONT.LABEL)
  # myfigs2 = ggarrange(plotlist = read_list(item_fig_list[2:7]),
  #   labels = labels, ncol=3, nrow=2, font.label = FONT.LABEL)
  # ggsave(filename = file.path(OUT_DIR, figname), 
  #   plot = myfigs,
  #   dpi = DPI,
  #   width = WIDTH*ncol,   
  #   height = HEIGHT*nrow,
  #   units = "in")
  
  save_fig("Fig_Items_notenc.jpg", item_notenc_fig_list, nrow=2, ncol=5)
  
  save_fig("Fig_Item_PCA.jpg", item_fig_list[1], nrow=1, ncol=1, wlabel=F)
#  save_fig("Fig_Items.pdf", item_fig_list[2:11], nrow=2, ncol=5)
  save_fig("Fig_Items.jpg", item_fig_list[2:7], nrow=2, ncol=3)
  save_fig("Fig_Distance.jpg", distances_fig_list, nrow=2, ncol=3)
  save_fig("Fig_Cluster.jpg", cluster_fig_list, nrow=2, ncol=2)
  save_fig("Fig_ClusterSupp1.jpg", cluster_supp_fig_list1, nrow=1, ncol=2)
  #save_fig("Fig_ClusterSupp2.png", cluster_supp_fig_list2, nrow=2, ncol=1)
  #save_fig("Fig_Students.png", student_fig_list, nrow=2, ncol=3)
  #save_fig("Fig_Edges.png", edge_fig_list, nrow=2, ncol=2)
  
}

########################
# Psychoco 2023 Figures
########################

OUT_DIR = file.path(FIG_DIR,  'psychoco23') 

if (F) {
  
  item.scales_fig_list = list(
    'A' = 'SAGE_full_items/type_sct-x-y-var_domain.png',
    'B' = 'SAGE_full_items/type_sct-z-y-var_domain.png',
    'C' = 'SAGE_full_items/type_sct-x-y-var_scale.png',
    'D' = 'SAGE_full_items/type_sct-z-y-var_scale.png'
  )
  
  item.difficulty_fig_list = list(
    'A' = 'SAGE_full_items/type_sct-x-y-var_IRT_difficulty.png',
    'B' = 'SAGE_full_items/type_sct-z-y-var_IRT_difficulty.png',
    'C' = 'SAGE_full_dim_items/type_reg-x-None-var_IRT_difficulty.png'
  )
  
  cluster_fig_list = list(
    'A' = 'SAGE_full_bw_random_CH.png'
  )
  
  student.gender_fig_list = list(
    'A' = 'SAGE_full_students_PCA.png',
    'B' = 'SAGE_full_students/type_sct-x-y-var_Gender.png',
    'C' = 'SAGE_full_students/type_sct-z-y-var_Gender.png',
    'D' = 'SAGE_full_dim_students/type_kde-x-None-var_Gender.png',
    'E' = 'SAGE_full_dim_students/type_kde-y-None-var_Gender.png',
    'F' = 'SAGE_full_dim_students/type_kde-z-None-var_Gender.png'
  )
  
  student.motherTongue_fig_list = list(
    'A' = 'SAGE_full_students/type_sct-x-y-var_motherTongue.png',
    'B' = 'SAGE_full_students/type_sct-x-y-var_motherTongue.png',
    'C' = 'SAGE_full_students/type_sct-z-y-var_motherTongue.png',
    'D' = 'SAGE_full_dim_students/type_kde-x-None-var_motherTongue.png',
    'E' = 'SAGE_full_dim_students/type_kde-y-None-var_motherTongue.png',
    'F' = 'SAGE_full_dim_students/type_kde-z-None-var_motherTongue.png'
  )
  
  edge_fig_list = list(
    #'A' = 'SAGE_full_edges_PCA.png',
    'A' = 'SAGE_full_age_edges_agg_wl.png',
    'B' = 'SAGE_full_ability_edges_agg_wl.png',
    'C' = 'SAGE_full_previous_sessions_edges.png',
    'D' = 'SAGE_full_frequency_edges.png'
  )
  
  save_fig("Fig_Students.gender.png", student.gender_fig_list, nrow=2, ncol=3, wlabels=F)
  save_fig("Fig_Students.tongue.png", student.motherTongue_fig_list, nrow=2, ncol=3, wlabels=F)
  save_fig("Fig_Cluster.png", cluster_fig_list, nrow=1, ncol=1, wlabels=F)
  save_fig("Fig_Items.scales.png", item.scales_fig_list, nrow=2, ncol=2, wlabels=F)
  save_fig("Fig_Items.difficulty.png", item.difficulty_fig_list, nrow=1, ncol=3, wlabels=F)
  save_fig("Fig_Edges.png", edge_fig_list, nrow=2, ncol=2, wlabels=F)
  
}
