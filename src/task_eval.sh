# CUDA_VISIBLE_DEVICES=1, nohup python src/task_eval_set.py --ckpt_name "set6_en_finetune_null_0.25_1.0e-05_r8" --ckpt_id "12268" --eval_lang "set6_hi" --is_zero_shot 1 --intervene_by "mean_mu_act" > hi-eval1.out & 
# CUDA_VISIBLE_DEVICES=1, nohup python src/task_eval_set.py --ckpt_name "set6_en_finetune_en_0.25_1.0e-05_r8" --ckpt_id "12268" --eval_lang "set6_hi" --is_zero_shot 1 --intervene_by "mean_mu_act" > hi-eval2.out & 
# CUDA_VISIBLE_DEVICES=1, nohup python src/task_eval_set.py --ckpt_name "set6_en_finetune_hi_0.25_1.0e-05_r8" --ckpt_id "12268" --eval_lang "set6_hi" --is_zero_shot 1 --intervene_by "mean_mu_act" > hi-eval3.out & 
# CUDA_VISIBLE_DEVICES=1, nohup python src/task_eval_set.py --ckpt_name "set6_en_finetune_en+hi_0.25_1.0e-05_r8" --ckpt_id "12268" --eval_lang "set6_hi" --is_zero_shot 1 --intervene_by "mean_mu_act" > hi-eval4.out &  

# CUDA_VISIBLE_DEVICES=1, nohup python src/task_eval_set.py --ckpt_name "set6_en_finetune_null_0.25_1.0e-05_r8" --ckpt_id "12268" --eval_lang "set6_hi" --is_zero_shot 0 --intervene_by "mean_p95_act" > hi-eval9.out & 
# CUDA_VISIBLE_DEVICES=2, nohup python src/task_eval_set.py --ckpt_name "set6_en_finetune_en_0.25_1.0e-05_r8" --ckpt_id "12268" --eval_lang "set6_hi" --is_zero_shot 0 --intervene_by "mean_p95_act" > hi-eval10.out & 
# CUDA_VISIBLE_DEVICES=3, nohup python src/task_eval_set.py --ckpt_name "set6_en_finetune_hi_0.25_1.0e-05_r8" --ckpt_id "12268" --eval_lang "set6_hi" --is_zero_shot 0 --intervene_by "mean_p95_act" > hi-eval11.out & 
# CUDA_VISIBLE_DEVICES=4, nohup python src/task_eval_set.py --ckpt_name "set6_en_finetune_en+hi_0.25_1.0e-05_r8" --ckpt_id "12268" --eval_lang "set6_hi" --is_zero_shot 0 --intervene_by "mean_p95_act" > hi-eval12.out &  

# CUDA_VISIBLE_DEVICES=1, nohup python src/task_eval_set.py --ckpt_name "set6_en_finetune_null_0.25_1.0e-05_r8" --ckpt_id "12268" --eval_lang "set6_ur" --is_zero_shot 0 --intervene_by "mean_p95_act" > ur-eval9.out & 
# CUDA_VISIBLE_DEVICES=2, nohup python src/task_eval_set.py --ckpt_name "set6_en_finetune_en_0.25_1.0e-05_r8" --ckpt_id "12268" --eval_lang "set6_ur" --is_zero_shot 0 --intervene_by "mean_p95_act" > ur-eval10.out & 
# CUDA_VISIBLE_DEVICES=2, nohup python src/task_eval_set.py --ckpt_name "set6_en_finetune_ur_0.25_1.0e-05_r8" --ckpt_id "12268" --eval_lang "set6_ur" --is_zero_shot 0 --intervene_by "mean_p95_act" > ur-eval11.out & 
# CUDA_VISIBLE_DEVICES=4, nohup python src/task_eval_set.py --ckpt_name "set6_en_finetune_en+ur_0.25_1.0e-05_r8" --ckpt_id "12268" --eval_lang "set6_ur" --is_zero_shot 0 --intervene_by "mean_p95_act" > ur-eval12.out &  

# CUDA_VISIBLE_DEVICES=2, nohup python src/task_eval_set.py --ckpt_name "set1_en_finetune_null_0.25_1.0e-05_r8" --ckpt_id "12268" --eval_lang "set1_vi" --is_zero_shot 0 --intervene_by "mean_p75_act" > eval5.out & 
# CUDA_VISIBLE_DEVICES=2, nohup python src/task_eval_set.py --ckpt_name "set1_en_finetune_en_0.25_1.0e-05_r8" --ckpt_id "12268" --eval_lang "set1_vi" --is_zero_shot 0 --intervene_by "mean_p75_act" > eval6.out & 
# CUDA_VISIBLE_DEVICES=2, nohup python src/task_eval_set.py --ckpt_name "set1_en_finetune_vi_0.25_1.0e-05_r8" --ckpt_id "12268" --eval_lang "set1_vi" --is_zero_shot 0 --intervene_by "mean_p75_act" > eval7.out & 
# CUDA_VISIBLE_DEVICES=2, nohup python src/task_eval_set.py --ckpt_name "set1_en_finetune_en+vi_0.25_1.0e-05_r8" --ckpt_id "12268" --eval_lang "set1_vi" --is_zero_shot 0 --intervene_by "mean_p75_act" > eval8.out & 

# CUDA_VISIBLE_DEVICES=2, nohup python src/task_eval_set.py --ckpt_name "set1_en_finetune_null_0.25_1.0e-05_r8" --ckpt_id "12268" --eval_lang "set1_vi" --is_zero_shot 0 --intervene_by "mean_p90_act" > eval9.out & 
# CUDA_VISIBLE_DEVICES=2, nohup python src/task_eval_set.py --ckpt_name "set1_en_finetune_en_0.25_1.0e-05_r8" --ckpt_id "12268" --eval_lang "set1_vi" --is_zero_shot 0 --intervene_by "mean_p90_act" > eval10.out & 
# CUDA_VISIBLE_DEVICES=2, nohup python src/task_eval_set.py --ckpt_name "set1_en_finetune_vi_0.25_1.0e-05_r8" --ckpt_id "12268" --eval_lang "set1_vi" --is_zero_shot 0 --intervene_by "mean_p90_act" > eval11.out & 
# CUDA_VISIBLE_DEVICES=2, nohup python src/task_eval_set.py --ckpt_name "set1_en_finetune_en+vi_0.25_1.0e-05_r8" --ckpt_id "12268" --eval_lang "set1_vi" --is_zero_shot 0 --intervene_by "mean_p90_act" > eval12.out & 

# CUDA_VISIBLE_DEVICES=2, nohup python src/task_eval_set.py --ckpt_name "set1_en_finetune_null_0.25_1.0e-05_r8" --ckpt_id "12268" --eval_lang "set1_vi" --is_zero_shot 0 --intervene_by "mean_p95_act" > eval13.out & 
# CUDA_VISIBLE_DEVICES=2, nohup python src/task_eval_set.py --ckpt_name "set1_en_finetune_en_0.25_1.0e-05_r8" --ckpt_id "12268" --eval_lang "set1_vi" --is_zero_shot 0 --intervene_by "mean_p95_act" > eval14.out & 
# CUDA_VISIBLE_DEVICES=2, nohup python src/task_eval_set.py --ckpt_name "set1_en_finetune_vi_0.25_1.0e-05_r8" --ckpt_id "12268" --eval_lang "set1_vi" --is_zero_shot 0 --intervene_by "mean_p95_act" > eval15.out & 
# CUDA_VISIBLE_DEVICES=2, nohup python src/task_eval_set.py --ckpt_name "set1_en_finetune_en+vi_0.25_1.0e-05_r8" --ckpt_id "12268" --eval_lang "set1_vi" --is_zero_shot 0 --intervene_by "mean_p95_act" > eval16.out & 

CUDA_VISIBLE_DEVICES=6, nohup python src/task_eval.py --ckpt_name "act_prob_90p_en_finetune_vi_0.25_1.0e-05_r8" --ckpt_id "12268" --eval_lang "vi" --is_zero_shot 1 --intervene_by "mean_mu_act" > eval1.out & 
CUDA_VISIBLE_DEVICES=6, nohup python src/task_eval.py --ckpt_name "act_prob_90p_en_finetune_vi_0.25_1.0e-05_r8" --ckpt_id "12268" --eval_lang "vi" --is_zero_shot 0 --intervene_by "mean_p75_act" > eval2.out & 
CUDA_VISIBLE_DEVICES=6, nohup python src/task_eval.py --ckpt_name "act_prob_90p_en_finetune_vi_0.25_1.0e-05_r8" --ckpt_id "12268" --eval_lang "vi" --is_zero_shot 0 --intervene_by "mean_p90_act" > eval3.out & 
CUDA_VISIBLE_DEVICES=6, nohup python src/task_eval.py --ckpt_name "act_prob_90p_en_finetune_vi_0.25_1.0e-05_r8" --ckpt_id "12268" --eval_lang "vi" --is_zero_shot 0 --intervene_by "mean_p95_act" > eval4.out & 

CUDA_VISIBLE_DEVICES=6, nohup python src/task_eval.py --ckpt_name "act_prob_90p_en_finetune_en+vi_0.25_1.0e-05_r8" --ckpt_id "12268" --eval_lang "vi" --is_zero_shot 1 --intervene_by "mean_mu_act" > eval5.out & 
CUDA_VISIBLE_DEVICES=6, nohup python src/task_eval.py --ckpt_name "act_prob_90p_en_finetune_en+vi_0.25_1.0e-05_r8" --ckpt_id "12268" --eval_lang "vi" --is_zero_shot 0 --intervene_by "mean_p75_act" > eval6.out & 
CUDA_VISIBLE_DEVICES=6, nohup python src/task_eval.py --ckpt_name "act_prob_90p_en_finetune_en+vi_0.25_1.0e-05_r8" --ckpt_id "12268" --eval_lang "vi" --is_zero_shot 0 --intervene_by "mean_p90_act" > eval7.out & 
CUDA_VISIBLE_DEVICES=6, nohup python src/task_eval.py --ckpt_name "act_prob_90p_en_finetune_en+vi_0.25_1.0e-05_r8" --ckpt_id "12268" --eval_lang "vi" --is_zero_shot 0 --intervene_by "mean_p95_act" > eval8.out & 
