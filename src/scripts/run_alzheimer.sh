for concept in pathway GO
do
    for fold in 0 1 2 3 4
    do
        python run.py \
            --data alzheimer \
            --concept $concept \
            --n_proto 13 \
            --train_batch_size 256 \
            --test_batch_size 256 \
            --lr 8e-4 \
            --h_dim 64 \
            --z_dim 32 \
            --max_epoch 150 \
            --model DeepGSEA \
            --exp_str 'optimal_'$fold \
            --device cuda:0 \
            --lambda_1 0.1 \
            --lambda_3 0.1\
            --lambda_4 1 \
            --lambda_5 1 \
            --ratio 0.8 \
            --fold $fold \
            --use_label
    done
done

for concept in GO
do
    for fold in 0
    do
        python interpret.py \
            --interpret label \
            --concept_id GO:0048699 \
            --data alzheimer \
            --concept $concept \
            --n_proto 13 \
            --train_batch_size 256 \
            --test_batch_size 256 \
            --lr 8e-4 \
            --h_dim 64 \
            --z_dim 32 \
            --max_epoch 150 \
            --model DeepGSEA \
            --exp_str 'optimal_'$fold \
            --device cuda:0 \
            --lambda_1 0.1 \
            --lambda_3 0.1\
            --lambda_4 1 \
            --lambda_5 1 \
            --ratio 0.8 \
            --fold $fold \
            --use_label
    done
done

for concept in GO
do
    for fold in 0
    do
        python interpret.py \
            --interpret phenotype \
            --concept_id GO:0048699 \
            --data alzheimer \
            --concept $concept \
            --n_proto 13 \
            --train_batch_size 256 \
            --test_batch_size 256 \
            --lr 8e-4 \
            --h_dim 64 \
            --z_dim 32 \
            --max_epoch 150 \
            --model DeepGSEA \
            --exp_str 'optimal_'$fold \
            --device cuda:0 \
            --lambda_1 0.1 \
            --lambda_3 0.1\
            --lambda_4 1 \
            --lambda_5 1 \
            --ratio 0.8 \
            --fold $fold \
            --use_label
    done
done

for concept in GO
do
    for fold in 0
    do
        python interpret.py \
            --interpret prediction \
            --concept_id GO:0048699 \
            --data alzheimer \
            --concept $concept \
            --n_proto 13 \
            --train_batch_size 256 \
            --test_batch_size 256 \
            --lr 8e-4 \
            --h_dim 64 \
            --z_dim 32 \
            --max_epoch 150 \
            --model DeepGSEA \
            --exp_str 'optimal_'$fold \
            --device cuda:0 \
            --lambda_1 0.1 \
            --lambda_3 0.1\
            --lambda_4 1 \
            --lambda_5 1 \
            --ratio 0.8 \
            --fold $fold \
            --use_label \
            --element_wise
    done
done