export HF_ENDPOINT=https://hf-mirror.com
# pretrain
huggingface-cli download --repo-type dataset pleisto/wikipedia-cn-20230720-filtered --local-dir datasets/pleisto/wikipedia-cn-20230720-filtered
huggingface-cli download --repo-type dataset xuqinyang/BaiduBaike-5.63M --local-dir datasets/xuqinyang/BaiduBaike-5.63M
huggingface-cli download --repo-type dataset wangrui6/Zhihu-KOL --local-dir datasets/wangrui6/Zhihu-KOL
huggingface-cli download --repo-type dataset wdndev/webnovel-chinese --local-dir datasets/wdndev/webnovel-chinese
huggingface-cli download --repo-type dataset TigerResearch/pretrain_zh --local-dir datasets/TigerResearch/pretrain_zh

# sft
huggingface-cli download --repo-type dataset BelleGroup/train_2M_CN --local-dir datasets/BelleGroup/train_2M_CN
huggingface-cli download --repo-type dataset YeungNLP/firefly-train-1.1M --local-dir datasets/YeungNLP/firefly-train-1.1M

huggingface-cli download --repo-type dataset TigerResearch/sft_zh --local-dir datasets/TigerResearch/pretrain_zh

