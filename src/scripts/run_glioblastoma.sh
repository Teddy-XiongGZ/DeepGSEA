for concept in GO pathway
do
    for fold in 0 1 2 3 4
    do
        python run.py \
            --data glioblastoma \
            --concept $concept \
            --n_proto 1 \
            --train_batch_size 64 \
            --test_batch_size 64 \
            --test_step 5 \
            --lr 1e-3 \
            --h_dim 64 \
            --z_dim 32 \
            --max_epoch 50 \
            --model DeepGSEA \
            --exp_str 'optimal_'$fold \
            --device cuda:0 \
            --lambda_1 0.1 \
            --lambda_4 1 \
            --lambda_5 10 \
            --ratio 0.8 \
            --fold $fold
    done
done

for concept in GO
do
    for fold in 0
    do
        python interpret.py \
            --interpret phenotype \
            --concept_id GO:0046717 \
            --data glioblastoma \
            --concept $concept \
            --n_proto 1 \
            --train_batch_size 64 \
            --test_batch_size 64 \
            --test_step 5 \
            --lr 1e-3 \
            --h_dim 64 \
            --z_dim 32 \
            --max_epoch 50 \
            --model DeepGSEA \
            --exp_str 'optimal_'$fold \
            --device cuda:0 \
            --lambda_1 0.1 \
            --lambda_4 1 \
            --lambda_5 10 \
            --ratio 0.8 \
            --fold $fold
    done
done

for concept in pathway
do
    for fold in 0
    do
        python interpret.py \
            --interpret phenotype \
            --concept_id R-HSA-444821 R-HSA-8866427 \
            --data glioblastoma \
            --concept $concept \
            --n_proto 1 \
            --train_batch_size 64 \
            --test_batch_size 64 \
            --test_step 5 \
            --lr 1e-3 \
            --h_dim 64 \
            --z_dim 32 \
            --max_epoch 50 \
            --model DeepGSEA \
            --exp_str 'optimal_'$fold \
            --device cuda:0 \
            --lambda_1 0.1 \
            --lambda_4 1 \
            --lambda_5 10 \
            --ratio 0.8 \
            --fold $fold
    done
done

for concept in GO
do
    for fold in 0
    do
        python interpret.py \
            --interpret heatmap \
            --concept_id GO:0046717 \
            --data glioblastoma \
            --concept $concept \
            --n_proto 1 \
            --train_batch_size 64 \
            --test_batch_size 64 \
            --test_step 5 \
            --lr 1e-3 \
            --h_dim 64 \
            --z_dim 32 \
            --max_epoch 50 \
            --model DeepGSEA \
            --exp_str 'optimal_'$fold \
            --device cuda:0 \
            --lambda_1 0.1 \
            --lambda_4 1 \
            --lambda_5 10 \
            --ratio 0.8 \
            --fold $fold
    done
done