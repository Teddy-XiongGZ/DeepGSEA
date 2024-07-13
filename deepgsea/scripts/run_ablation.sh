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
            --lr 1e-4 \
            --h_dim 64 \
            --z_dim 32 \
            --max_epoch 150 \
            --model DeepGSEA \
            --exp_str 'no_label_'$fold \
            --device cuda:0 \
            --lambda_1 0.1 \
            --lambda_3 1\
            --lambda_4 1 \
            --lambda_5 1 \
            --ratio 0.8 \
            --fold $fold

        python interpret.py \
            --interpret phenotype \
            --data alzheimer \
            --concept $concept \
            --n_proto 13 \
            --train_batch_size 256 \
            --test_batch_size 256 \
            --lr 1e-4 \
            --h_dim 64 \
            --z_dim 32 \
            --max_epoch 150 \
            --model DeepGSEA \
            --exp_str 'no_label_'$fold \
            --device cuda:0 \
            --lambda_1 0.1 \
            --lambda_3 1\
            --lambda_4 1 \
            --lambda_5 1 \
            --ratio 0.8 \
            --fold $fold

        python interpret.py \
            --interpret label \
            --data alzheimer \
            --concept $concept \
            --n_proto 13 \
            --train_batch_size 256 \
            --test_batch_size 256 \
            --lr 1e-4 \
            --h_dim 64 \
            --z_dim 32 \
            --max_epoch 150 \
            --model DeepGSEA \
            --exp_str 'no_label_'$fold \
            --device cuda:0 \
            --lambda_1 0.1 \
            --lambda_3 1\
            --lambda_4 1 \
            --lambda_5 1 \
            --ratio 0.8 \
            --fold $fold
    done
done

for concept in pathway GO
do
    for fold in 0 1 2 3 4
    do
        python run.py \
            --data alzheimer \
            --concept $concept \
            --n_proto 1 \
            --train_batch_size 256 \
            --test_batch_size 256 \
            --lr 1e-4 \
            --h_dim 64 \
            --z_dim 32 \
            --max_epoch 150 \
            --model DeepGSEA \
            --exp_str 'one_proto_'$fold \
            --device cuda:0 \
            --lambda_1 0.1 \
            --lambda_4 1 \
            --lambda_5 1 \
            --ratio 0.8 \
            --fold $fold

        python interpret.py \
            --interpret phenotype \
            --data alzheimer \
            --concept $concept \
            --n_proto 1 \
            --train_batch_size 256 \
            --test_batch_size 256 \
            --lr 1e-4 \
            --h_dim 64 \
            --z_dim 32 \
            --max_epoch 150 \
            --model DeepGSEA \
            --exp_str 'one_proto_'$fold \
            --device cuda:0 \
            --lambda_1 0.1 \
            --lambda_4 1 \
            --lambda_5 1 \
            --ratio 0.8 \
            --fold $fold

        python interpret.py \
            --interpret label \
            --data alzheimer \
            --concept $concept \
            --n_proto 1 \
            --train_batch_size 256 \
            --test_batch_size 256 \
            --lr 1e-4 \
            --h_dim 64 \
            --z_dim 32 \
            --max_epoch 150 \
            --model DeepGSEA \
            --exp_str 'one_proto_'$fold \
            --device cuda:0 \
            --lambda_1 0.1 \
            --lambda_4 1 \
            --lambda_5 1 \
            --ratio 0.8 \
            --fold $fold
    done
done

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
            --exp_str 'one_stage_'$fold \
            --device cuda:0 \
            --lambda_1 0.1 \
            --lambda_3 0.1\
            --lambda_4 1 \
            --lambda_5 1 \
            --ratio 0.8 \
            --fold $fold \
            --use_label \
            --one_step

        python interpret.py \
            --interpret phenotype \
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
            --exp_str 'one_stage_'$fold \
            --device cuda:0 \
            --lambda_1 0.1 \
            --lambda_3 0.1\
            --lambda_4 1 \
            --lambda_5 1 \
            --fold $fold \
            --use_label \
            --one_step

        python interpret.py \
            --interpret label \
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
            --exp_str 'one_stage_'$fold \
            --device cuda:0 \
            --lambda_1 0.1 \
            --lambda_3 0.1\
            --lambda_4 1 \
            --lambda_5 1 \
            --fold $fold \
            --use_label \
            --one_step
    done
done

for concept in ONE
do
    for fold in 0 1 2 3 4
    do
        python run.py \
            --data alzheimer \
            --concept $concept \
            --n_proto 13 \
            --train_batch_size 256 \
            --test_batch_size 256 \
            --lr 2e-4 \
            --h_dim 64 \
            --z_dim 32 \
            --max_epoch 150 \
            --model DeepGSEA \
            --exp_str 'one_gene_set_'$fold \
            --device cuda:0 \
            --lambda_1 100 \
            --lambda_3 100\
            --lambda_4 1 \
            --lambda_5 1 \
            --ratio 0.8 \
            --fold $fold \
            --use_label

        python interpret.py \
            --interpret phenotype \
            --data alzheimer \
            --concept $concept \
            --n_proto 13 \
            --train_batch_size 256 \
            --test_batch_size 256 \
            --lr 2e-4 \
            --h_dim 64 \
            --z_dim 32 \
            --max_epoch 150 \
            --model DeepGSEA \
            --exp_str 'one_gene_set_'$fold \
            --device cuda:0 \
            --lambda_1 100 \
            --lambda_3 100\
            --lambda_4 1 \
            --lambda_5 1 \
            --fold $fold \
            --use_label

        python interpret.py \
            --interpret label \
            --data alzheimer \
            --concept $concept \
            --n_proto 13 \
            --train_batch_size 256 \
            --test_batch_size 256 \
            --lr 2e-4 \
            --h_dim 64 \
            --z_dim 32 \
            --max_epoch 150 \
            --model DeepGSEA \
            --exp_str 'one_gene_set_'$fold \
            --device cuda:0 \
            --lambda_1 100 \
            --lambda_3 100\
            --lambda_4 1 \
            --lambda_5 1 \
            --fold $fold \
            --use_label
    done
done

for downsample in '500'
do
    for n_proto in 2
    do
        for lambda_3 in 0 1 10 100
        do
            for lambda_4 in 1
            do
                for lambda_5 in 1
                do
                    for concept in pathway
                    do
                        for fold in 0
                        do
                            python run.py \
                                --data influenza \
                                --concept $concept \
                                --n_proto $n_proto \
                                --train_batch_size 64 \
                                --test_batch_size 64 \
                                --lr 1e-3 \
                                --h_dim 64 \
                                --z_dim 32 \
                                --max_epoch 100 \
                                --model DeepGSEA \
                                --exp_str $downsample'_'$n_proto'_'$lambda_3'_'$lambda_4'_'$lambda_5'_'$fold \
                                --device cuda:0 \
                                --lambda_1 0.1 \
                                --lambda_3 $lambda_3 \
                                --lambda_4 $lambda_4 \
                                --lambda_5 $lambda_5 \
                                --ratio 0.5 \
                                --fold $fold \
                                --downsample $downsample

                            python interpret.py \
                                --interpret phenotype \
                                --data influenza \
                                --concept_id R-HSA-72766 \
                                --concept $concept \
                                --n_proto $n_proto \
                                --train_batch_size 64 \
                                --test_batch_size 64 \
                                --lr 1e-3 \
                                --h_dim 64 \
                                --z_dim 32 \
                                --max_epoch 100 \
                                --model DeepGSEA \
                                --exp_str $downsample'_'$n_proto'_'$lambda_3'_'$lambda_4'_'$lambda_5'_'$fold \
                                --device cuda:0 \
                                --lambda_1 0.1 \
                                --lambda_3 $lambda_3 \
                                --lambda_4 $lambda_4 \
                                --lambda_5 $lambda_5 \
                                --ratio 0.5 \
                                --fold $fold \
                                --downsample $downsample
                        done
                    done
                done
            done
        done
    done
done

for downsample in '500'
do
    for n_proto in 2
    do
        for lambda_3 in 1
        do
            for lambda_4 in 0 1 10 100
            do
                for lambda_5 in 1
                do
                    for concept in pathway
                    do
                        for fold in 0
                        do
                            python run.py \
                                --data influenza \
                                --concept $concept \
                                --n_proto $n_proto \
                                --train_batch_size 64 \
                                --test_batch_size 64 \
                                --lr 1e-3 \
                                --h_dim 64 \
                                --z_dim 32 \
                                --max_epoch 100 \
                                --model DeepGSEA \
                                --exp_str $downsample'_'$n_proto'_'$lambda_3'_'$lambda_4'_'$lambda_5'_'$fold \
                                --device cuda:0 \
                                --lambda_1 0.1 \
                                --lambda_3 $lambda_3 \
                                --lambda_4 $lambda_4 \
                                --lambda_5 $lambda_5 \
                                --ratio 0.5 \
                                --fold $fold \
                                --downsample $downsample

                            python interpret.py \
                                --interpret phenotype \
                                --data influenza \
                                --concept_id R-HSA-72766 \
                                --concept $concept \
                                --n_proto $n_proto \
                                --train_batch_size 64 \
                                --test_batch_size 64 \
                                --lr 1e-3 \
                                --h_dim 64 \
                                --z_dim 32 \
                                --max_epoch 100 \
                                --model DeepGSEA \
                                --exp_str $downsample'_'$n_proto'_'$lambda_3'_'$lambda_4'_'$lambda_5'_'$fold \
                                --device cuda:0 \
                                --lambda_1 0.1 \
                                --lambda_3 $lambda_3 \
                                --lambda_4 $lambda_4 \
                                --lambda_5 $lambda_5 \
                                --ratio 0.5 \
                                --fold $fold \
                                --downsample $downsample
                        done
                    done
                done
            done
        done
    done
done

for downsample in '500'
do
    for n_proto in 2
    do
        for lambda_3 in 1
        do
            for lambda_4 in 1
            do
                for lambda_5 in 0 1 10 100
                do
                    for concept in pathway
                    do
                        for fold in 0
                        do
                            python run.py \
                                --data influenza \
                                --concept $concept \
                                --n_proto $n_proto \
                                --train_batch_size 64 \
                                --test_batch_size 64 \
                                --lr 1e-3 \
                                --h_dim 64 \
                                --z_dim 32 \
                                --max_epoch 100 \
                                --model DeepGSEA \
                                --exp_str $downsample'_'$n_proto'_'$lambda_3'_'$lambda_4'_'$lambda_5'_'$fold \
                                --device cuda:0 \
                                --lambda_1 0.1 \
                                --lambda_3 $lambda_3 \
                                --lambda_4 $lambda_4 \
                                --lambda_5 $lambda_5 \
                                --ratio 0.5 \
                                --fold $fold \
                                --downsample $downsample

                            python interpret.py \
                                --interpret phenotype \
                                --data influenza \
                                --concept_id R-HSA-72766 \
                                --concept $concept \
                                --n_proto $n_proto \
                                --train_batch_size 64 \
                                --test_batch_size 64 \
                                --lr 1e-3 \
                                --h_dim 64 \
                                --z_dim 32 \
                                --max_epoch 100 \
                                --model DeepGSEA \
                                --exp_str $downsample'_'$n_proto'_'$lambda_3'_'$lambda_4'_'$lambda_5'_'$fold \
                                --device cuda:0 \
                                --lambda_1 0.1 \
                                --lambda_3 $lambda_3 \
                                --lambda_4 $lambda_4 \
                                --lambda_5 $lambda_5 \
                                --ratio 0.5 \
                                --fold $fold \
                                --downsample $downsample
                        done
                    done
                done
            done
        done
    done
done