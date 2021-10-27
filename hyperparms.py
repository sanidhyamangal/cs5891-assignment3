from ddpg_mangalsv import DDPG
from logger import logger

actor_lr = [1e-1, 1e-2, 1e-3, 2e-3, 5e-2, 5e-3, 5e-4]
critic_lr = [1e-1, 1e-2, 1e-3, 2e-3, 5e-2, 5e-3, 5e-4]

hidden_states = [32, 64, 128, 256, 512]
batch_size = [64, 128, 256, 512, 1024]

for ac_lr in actor_lr:
    for c_lr in critic_lr:
        for h_state in hidden_states:
            for bs in batch_size:
                ddpg = DDPG(num_hidden_states=h_state,batch_size=bs, critic_lr=c_lr, actor_lr=ac_lr)

                logger.info(f"Training for parms, hidden_state:{h_state}, batch_size:{bs}, c_lr:{c_lr}, ac_lr:{ac_lr}")
                ddpg.train(300,plot_name=f"mc-{h_state}-{bs}-{ac_lr}-{c_lr}.png")