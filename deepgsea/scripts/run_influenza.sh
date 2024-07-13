for concept in pathway GO
do
    for fold in 0 1 2 3 4
    do
        python run.py \
            --data influenza \
            --concept $concept \
            --n_proto 1 \
            --train_batch_size 64 \
            --test_batch_size 64 \
            --lr 1e-3 \
            --h_dim 64 \
            --z_dim 32 \
            --max_epoch 100 \
            --model DeepGSEA \
            --exp_str 'optimal_'$fold \
            --device cuda:0 \
            --lambda_1 0.1 \
            --lambda_4 1 \
            --lambda_5 1 \
            --ratio 0.5 \
            --fold $fold \
            --downsample '500'
    done
done

for concept in pathway
do
    for fold in 0
    do
        python interpret.py \
            --interpret phenotype_origin \
            --concept_id R-HSA-72766 \
            --data influenza \
            --concept $concept \
            --n_proto 1 \
            --train_batch_size 64 \
            --test_batch_size 64 \
            --lr 1e-3 \
            --h_dim 64 \
            --z_dim 32 \
            --max_epoch 100 \
            --model DeepGSEA \
            --exp_str 'optimal_'$fold \
            --device cuda:0 \
            --lambda_1 0.1 \
            --lambda_4 1 \
            --lambda_5 1 \
            --ratio 0.5 \
            --fold $fold \
            --downsample '500'
    done
done

for concept in pathway
do
    for fold in 0
    do
        python interpret.py \
            --interpret phenotype \
            --concept_id R-HSA-72766 R-HSA-5625900 \
            --data influenza \
            --concept $concept \
            --n_proto 1 \
            --train_batch_size 64 \
            --test_batch_size 64 \
            --lr 1e-3 \
            --h_dim 64 \
            --z_dim 32 \
            --max_epoch 100 \
            --model DeepGSEA \
            --exp_str 'optimal_'$fold \
            --device cuda:0 \
            --lambda_1 0.1 \
            --lambda_4 1 \
            --lambda_5 1 \
            --ratio 0.5 \
            --fold $fold \
            --downsample '500'
    done
done

for concept in pathway
do
    for fold in 0
    do
        python interpret.py \
            --interpret similarity \
            --concept_id R-HSA-72766 \
            --data influenza \
            --concept $concept \
            --n_proto 1 \
            --train_batch_size 64 \
            --test_batch_size 64 \
            --lr 1e-3 \
            --h_dim 64 \
            --z_dim 32 \
            --max_epoch 100 \
            --model DeepGSEA \
            --exp_str 'optimal_'$fold \
            --device cuda:0 \
            --lambda_1 0.1 \
            --lambda_4 1 \
            --lambda_5 1 \
            --ratio 0.5 \
            --fold $fold \
            --downsample '500'
    done
done