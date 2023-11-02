for control in 0 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95 1
do
    for fold in 0 1 2 3 4
    do
        python run.py \
            --data simul_1 \
            --concept HALLMARK \
            --train_batch_size 256 \
            --test_batch_size 256 \
            --test_step 5 \
            --lr 1e-3 \
            --h_dim 64 \
            --z_dim 32 \
            --max_epoch 50 \
            --model DeepGSEA \
            --exp_str 'optimal_'$fold \
            --device cuda:0 \
            --lambda_2 0.1 \
            --lambda_4 1 \
            --lambda_5 1 \
            --ratio 0.8 \
            --fold $fold \
            --control $control
    done
done

for control in 0 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95 1
do
    for fold in 0 1 2 3 4
    do
        python run.py \
            --data simul_2 \
            --concept HALLMARK \
            --train_batch_size 256 \
            --test_batch_size 256 \
            --test_step 5 \
            --lr 1e-3 \
            --h_dim 64 \
            --z_dim 32 \
            --max_epoch 50 \
            --model DeepGSEA \
            --exp_str 'optimal_'$fold \
            --device cuda:0 \
            --lambda_2 0.1 \
            --lambda_4 1 \
            --lambda_5 1 \
            --ratio 0.8 \
            --fold $fold \
            --control $control
    done
done


for control in 100 200 300 400 500 600 700 800 900 1000
do
    for fold in 0 1 2 3 4
    do
        python run.py \
            --data simul_3 \
            --concept HALLMARK \
            --train_batch_size 256 \
            --test_batch_size 256 \
            --lr 1e-3 \
            --h_dim 64 \
            --z_dim 32 \
            --max_epoch 50 \
            --model DeepGSEA \
            --exp_str 'optimal_'$fold \
            --device cuda:0 \
            --lambda_2 0.1 \
            --lambda_4 1 \
            --lambda_5 1 \
            --ratio 0.8 \
            --fold $fold \
            --control $control
    done
done

for control in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
do
    for fold in 0 1 2 3 4
    do
        python run.py \
            --data simul_4 \
            --concept HALLMARK \
            --train_batch_size 256 \
            --test_batch_size 256 \
            --lr 1e-3 \
            --h_dim 64 \
            --z_dim 32 \
            --max_epoch 50 \
            --model DeepGSEA \
            --exp_str 'optimal_'$fold \
            --device cuda:0 \
            --lambda_2 0.1 \
            --lambda_4 1 \
            --lambda_5 1 \
            --ratio 0.8 \
            --fold $fold \
            --control $control
    done
done