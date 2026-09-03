import numpy as np
import typer

def main(
        #stat: str = typer.Option(...),
        q: str = typer.Option(...),
):
    mepath = f'/home/he266/research/results/tables/en_stats/recon/ME/{q}.npz'
    fmpath = f'/home/he266/research/results/tables/en_stats/recon/FM/{q}.npz'

    statsfm = dict(np.load(fmpath))
    statsme = dict(np.load(mepath))

    print(f'For q:{q}')
    #print(f'ME {stat.replace('_', ' ')}={statsme[stat]} vs FM {stat.replace('_', ' ')}={statsfm[stat]}')
    print('FM:')
    print(statsfm)
    print('ME')
    print(statsme)

if __name__ == '__main__':
    typer.run(main)