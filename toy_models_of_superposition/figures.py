import matplotlib.pyplot as plt	
import numpy as np
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots



def plot_intro_diagram(model):
	from matplotlib import colors  as mcolors
	from matplotlib import collections  as mc
	WA = model.W.detach()
	N = len(WA[:,0])
	config = model.config
	sel = range(config.n_instances) # can be used to highlight specific sparsity levels
	if model.importance.numel() > 1:
		plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.viridis(model.importance[0].cpu().numpy()))
	plt.rcParams['figure.dpi'] = 200
	fig, axs = plt.subplots(1,len(sel), figsize=(2*len(sel),2))
	for i, ax in zip(sel, axs):
		W = WA[i].cpu().detach().numpy()
		colors = [mcolors.to_rgba(c)
			for c in plt.rcParams['axes.prop_cycle'].by_key()['color']]
		ax.scatter(W[:,0], W[:,1], c=colors[0:len(W[:,0])])
		ax.set_aspect('equal')
		ax.add_collection(mc.LineCollection(np.stack((np.zeros_like(W),W), axis=1), colors=colors))

		z = 1.5
		ax.set_facecolor('#FCFBF8')
		ax.set_xlim((-z,z))
		ax.set_ylim((-z,z))
		ax.tick_params(left = True, right = False , labelleft = False ,
					labelbottom = False, bottom = True)
		for spine in ['top', 'right']:
			ax.spines[spine].set_visible(False)
		for spine in ['bottom','left']:
			ax.spines[spine].set_position('center')
	plt.show()



def render_features(model, which=np.s_[:], to_sort=False):
	cfg = model.config
	W = model.W.detach()
	W_norm = W / (1e-5 + torch.linalg.norm(W, 2, dim=-1, keepdim=True))

	interference = torch.einsum('ifh,igh->ifg', W_norm, W)
	interference[:, torch.arange(cfg.n_features), torch.arange(cfg.n_features)] = 0

	polysemanticity = torch.linalg.norm(interference, dim=-1).cpu()
	net_interference = (interference**2 * model.feature_probability[:, None, :]).sum(-1).cpu()
	norms = torch.linalg.norm(W, 2, dim=-1).cpu()

	WtW = torch.einsum('sih,soh->sio', W, W).cpu()

	# width = weights[0].cpu()
	# x = torch.cumsum(width+0.1, 0) - width[0]
	x = torch.arange(cfg.n_features)
	width = 0.9

	which_instances = np.arange(cfg.n_instances)[which]
	fig = make_subplots(rows=len(which_instances),
						cols=2,
						shared_xaxes=True,
						vertical_spacing=0.02,
						horizontal_spacing=0.1)
	for (row, inst) in enumerate(which_instances):
		y = norms[inst]
		y = sorted(y, reverse=True) if to_sort else y
		fig.add_trace(
			go.Bar(x=x, 
				y=y,
				marker=dict(
					color=polysemanticity[inst],
					cmin=0,
					cmax=1
				),
				width=width,
			),
			row=1+row, col=1
		)
		data = WtW[inst].numpy()
		fig.add_trace(
			go.Image(
				z=plt.cm.coolwarm((1 + data)/2, bytes=True),
				colormodel='rgba256',
				customdata=data,
				hovertemplate='''\
		In: %{x}<br>
		Out: %{y}<br>
		Weight: %{customdata:0.2f}
		'''            
			),
			row=1+row, col=2
		)

	fig.add_vline(
	x=(x[cfg.n_hidden-1]+x[cfg.n_hidden])/2, 
	line=dict(width=0.5),
	col=1,
	)

	# fig.update_traces(marker_size=1)
	fig.update_layout(showlegend=False, 
					width=600,
					height=100*len(which_instances),
					margin=dict(t=0, b=0))
	fig.update_xaxes(visible=False)
	fig.update_yaxes(visible=False)
	return fig


@torch.no_grad()
def compute_dimensionality(W):
    norms = torch.linalg.norm(W, 2, dim=-1) 
    W_unit = W / torch.clamp(norms[:, :, None], 1e-6, float('inf'))

    interferences = (torch.einsum('eah,ebh->eab', W_unit, W)**2).sum(-1)

    dim_fracs = (norms**2/interferences)
    return dim_fracs.cpu()


def dimensionality_figure(model):
    fig = go.Figure()
    
    density = model.feature_probability[:, 0].cpu()
    W = model.W.detach()
    
    for a,b in [(1,2), (2,3), (2,5), (2,6), (2,7)]:
        val = a/b
        fig.add_hline(val, line_color="purple", opacity=0.2, annotation=dict(text=f"{a}/{b}"))
    
    for a,b in [(5,6), (4,5), (3,4), (3,8), (3,12), (3,20)]:
        val = a/b
        fig.add_hline(val, line_color="blue", opacity=0.2, annotation=dict(text=f"{a}/{b}", x=0.05))
    
    dim_fracs = compute_dimensionality(model.W)
    for i in range(len(W)):
        fracs_ = dim_fracs[i]
        N = fracs_.shape[0]
        xs = 1/density
        if i!= len(W)-1:
            dx = xs[i+1]-xs[i]
        fig.add_trace(
            go.Scatter(
                x=1/density[i]*np.ones(N)+dx*np.random.uniform(-0.1,0.1,N),
                y=fracs_,
                marker=dict(
                    color='black',
                    size=1.5,
                    opacity=0.5,
                ),
                mode='markers',
            )
        )
    
    fig.update_xaxes(
        type='log', 
        title='1/(1-S)',
        showgrid=False,
    )
    fig.update_yaxes(
        showgrid=False
    )
    fig.update_layout(showlegend=False)
    return fig