export HF_ENDPOINT=https://hf-mirror.com
# pretrain
huggingface-cli download --resume-download --repo-type dataset pleisto/wikipedia-cn-20230720-filtered --local-dir datasets/pretrain/pleisto/wikipedia-cn-20230720-filtered
huggingface-cli download --resume-download --repo-type dataset xuqinyang/BaiduBaike-5.63M --local-dir datasets/pretrain/xuqinyang/BaiduBaike-5.63M
huggingface-cli download --resume-download --repo-type dataset wangrui6/Zhihu-KOL --local-dir datasets/pretrain/wangrui6/Zhihu-KOL
huggingface-cli download --repo-type dataset wdndev/webnovel-chinese --local-dir datasets/pretrain/wdndev/webnovel-chinese
huggingface-cli download --resume-download --repo-type dataset TigerResearch/pretrain_zh --local-dir datasets/pretrain/TigerResearch/pretrain_zh

# sft
huggingface-cli download --repo-type dataset BelleGroup/train_2M_CN --local-dir datasets/sft/BelleGroup/train_2M_CN
huggingface-cli download --repo-type dataset YeungNLP/firefly-train-1.1M --local-dir datasets/sft/YeungNLP/firefly-train-1.1M
huggingface-cli download --repo-type dataset TigerResearch/sft_zh --local-dir datasets/sft/TigerResearch/pretrain_zh

# rlhf
huggingface-cli download --repo-type dataset mxz/CValues --local-dir datasets/rlhf/Cvalues
huggingface-cli download --repo-type dataset liyucheng/zhihu_rlhf_3k --local-dir datasets/rlhf/zhihu_rlhf_3k