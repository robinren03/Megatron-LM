
python tools/checkpoint/convert.py \
    --model-type GPT \
    --loader core \
    --saver megatron  \
    --load-dir /home/yylvsx/RL_Learn/345m_gpt \
    --save-dir /home/yylvsx/RL_Learn/345m_gpt_meg \
    --megatron-path /home/yylvsx/RL_Learn/Megatron-LM

