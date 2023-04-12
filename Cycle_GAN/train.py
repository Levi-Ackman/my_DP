import torch
from get_data import HorseZebraDataset
import sys
from utils import save_checkpoint, load_checkpoint,seed_everything
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import config
from tqdm import tqdm
from torchvision.utils import save_image
from Discriminator import Discriminator
from Generator import Generator


def train_fn(
    disc_H,
    disc_Z,
    gen_Z,
    gen_H,
    loader,
    opt_disc,
    opt_gen,
    l1,
    mse,
    d_scaler,
    g_scaler,
    ):
    ''' the two are used to document the train_process,i.e. compare the img_value between real and fake horse'''
    H_reals=0.0
    H_fakes=0.0
    loop=tqdm(loader,leave=True)

    for batch_idx,(zebra,horse) in enumerate(loop):
        zebra=zebra.to(config.DEVICE)
        horse=horse.to(config.DEVICE)

         # Train Discriminators H and Z
        with torch.cuda.amp.autocast():
            fake_horse=gen_H(zebra)  ## use zebra to gen horse
            ''' use D_H to judge whether is real horse or fake'''
            D_H_real=disc_H(horse)
            D_H_fake=disc_H(fake_horse.detach())
            H_reals+=D_H_real.mean().item()
            H_fakes+=D_H_fake.mean().item()

            D_H_real_loss=mse(D_H_real,torch.ones_like(D_H_real))  # make sure the real_img is judged as real, so we optim its value close to 1
            D_H_fake_loss=mse(D_H_fake,torch.zeros_like(D_H_fake))  # make sure the fake_img is judged as fake, so we optim its value close to 0
            D_H_loss=D_H_real_loss+D_H_fake_loss
            
            ''' the inverse process is much like the forward path we basically do the same thing in the opposite way'''
            fake_zebra=gen_Z(horse) #use horse to generate zebra
            D_Z_real=disc_Z(zebra)
            D_Z_fake=disc_Z(fake_zebra.detach())

            D_Z_real_loss=mse(D_Z_real,torch.ones_like(D_Z_real))  # make sure the real_img is judged as real, so we optim its value close to 1
            D_Z_fake_loss=mse(D_Z_fake,torch.zeros_like(D_Z_fake))  # make sure the fake_img is judged as fake, so we optim its value close to 0
            D_Z_loss=D_Z_real_loss+D_Z_fake_loss

            ## combine it together
            ## here we divide the D_loss by 2 cause we wanna the Dis and Gen are training in a reletively similar speed
            ## in case the Dis always outperform the Gen, that will break the training cycle!!
            D_loss=(D_H_loss+D_Z_loss)/2
        
        opt_disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train Generators H and Z
        with torch.cuda.amp.autocast():
            # adversarial loss for both generators
            D_H_fake=disc_H(fake_horse)
            D_Z_fake=disc_Z(fake_zebra)
            ''' the gen try to fool the dis so we wanna optim the fake_prob to 1'''
            loss_G_H=mse(D_H_fake,torch.ones_like(D_H_fake))
            loss_G_Z=mse(D_Z_fake,torch.ones_like(D_Z_fake))

            #cycle loss
            cycle_zebra=gen_Z(fake_horse)  #use fake_horse to gen zebra and judge its distance to real zebra
            cycle_horse=gen_Z(fake_zebra)  #use fake_zebra to gen horse and judge its distance to real horse
            cycle_zebra_loss=l1(zebra,cycle_zebra)
            cycle_horse_loss=l1(horse,cycle_horse)

            '''indentity loss(remove this when we use lambda_identity=0, i.e. when we are not train on oil_painting tasks)'''
            identity_zebra=gen_Z(zebra) #if we use zebra to gen zebra, then the gen shouldn't do much work
            identity_horse=gen_H(horse)  # the same when use horse to gen horse
            identity_zebra_loss=l1(zebra,identity_zebra)
            identity_horse_loss=l1(horse,identity_horse)

            # add them all together
            G_loss=(
                loss_G_Z
                +loss_G_H
                +cycle_zebra_loss*config.LAMBDA_CYCLE
                +cycle_horse_loss*config.LAMBDA_CYCLE
                +identity_zebra_loss*config.LAMBDA_IDENTITY
                +identity_horse_loss*config.LAMBDA_IDENTITY
            )
        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()
        
        if batch_idx % 1500 ==0:
           save_image(fake_horse*0.5 + 0.5,f'saved_images/fake_horse_{batch_idx}.png')
           save_image(fake_zebra*0.5 + 0.5,f'saved_images/fake_zebra_{batch_idx}.png')
        
        loop.set_postfix(H_real=H_reals/(batch_idx+1),H_fake=H_fakes/(batch_idx+1))
  

def main():
    disc_H=Discriminator(in_channels=3).to(config.DEVICE)
    disc_Z=Discriminator(in_channels=3).to(config.DEVICE)
    gen_Z = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    gen_H = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    ## we combine the two model with similar structure into one optimizer so the overall structure will be much more explicit
    ## and also to decrease the optimizer we used
    opt_disc=optim.Adam(
        list(disc_H.parameters()) + list(disc_Z.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5,0.999),
     )
    opt_gen=optim.Adam(
        list(gen_H.parameters())+list(gen_Z.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5,0.999),
     )

    L1_LOSS=nn.L1Loss()
    mse=nn.MSELoss()

    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_GEN_H,
            gen_H,
            opt_gen,
            config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_GEN_Z,
            gen_Z,
            opt_gen,
            config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_CRITIC_H,
            disc_H,
            opt_disc,
            config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_CRITIC_Z,
            disc_Z,
            opt_disc,
            config.LEARNING_RATE,
        )
    
    train_dataset=HorseZebraDataset(
        root_horse=config.HORSE_TRAIN_DIR,
        root_zebra=config.ZEBRA_TRAIN_DIR,
        transform=config.horse_zebra_transform
    )
    val_dataset=HorseZebraDataset(
        root_horse=config.HORSE_VAL_DIR,
        root_zebra=config.ZEBRA_VAL_DIR,
        transform=config.horse_zebra_transform
    )
    train_loader=DataLoader(
        dataset=train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )
    val_loader=DataLoader(
        dataset=val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )
    g_scaler=torch.cuda.amp.GradScaler()
    d_scaler=torch.cuda.amp.GradScaler()

    for epoch in range(config.NUM_WORKERS):
        train_fn(
            disc_H,
            disc_Z,
            gen_Z,
            gen_H,
            train_loader,
            opt_disc,
            opt_gen,
            L1_LOSS,
            mse,
            d_scaler,
            g_scaler,
        )
        if config.SAVE_MODEL:
            save_checkpoint(gen_H,opt_gen,filename=config.CHECKPOINT_GEN_H)
            save_checkpoint(gen_Z, opt_gen, filename=config.CHECKPOINT_GEN_Z)
            save_checkpoint(disc_H, opt_disc, filename=config.CHECKPOINT_CRITIC_H)
            save_checkpoint(disc_Z, opt_disc, filename=config.CHECKPOINT_CRITIC_Z)

if __name__=='__main__':
    main()
