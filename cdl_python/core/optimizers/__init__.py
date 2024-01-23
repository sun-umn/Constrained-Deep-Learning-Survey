from .pygranso_solver import pygranso_problem
from .tfco_solver import TFCO_problem

def init_optimizer(cfg,
                   data,
                   device,
                   model,
                   fn = ''):
    OPT_NAME = cfg.get('OPTIMIZER','NAME')
    if OPT_NAME == 'PyGRANSO':
        opt = pygranso_problem(
            cfg,data,device,model,fn=fn
        )
        return opt
    elif OPT_NAME == 'TFCO':
        opt = TFCO_problem(
                cfg,data,device,model,fn=fn
            )
        return opt
    else:
        raise ValueError(f"Optimizer <{OPT_NAME}> is not implemented")
    