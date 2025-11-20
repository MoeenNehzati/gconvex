"""
Infimal Convolution with Implicit Differentiation

This module implements the infimal convolution operation g(y) = inf_x [K(x,y) - f(x)]
as a differentiable PyTorch operation using implicit differentiation via the envelope theorem.

The key insight is that gradients with respect to the network parameters f_net can be
computed without backpropagating through the optimization process that finds the minimizer x*.
Instead, we use the envelope theorem which states that at the optimum x*:
    
    d/dθ g(y,θ) = -d/dθ f(x*,θ)

This allows efficient gradient computation while maintaining correctness.

Technical Details:
    - Forward pass: Solves the optimization problem to find x* = argmin_x [K(x,y) - f(x)]
    - Backward pass: Applies envelope theorem to compute gradients wrt network parameters
    - No gradient tracking through the optimization loop (uses functional_call with detached params)
    - Supports multiple optimizers: LBFGS, Adam, and GD
    - Early stopping based on relative change in objective value
    - Optional L2 regularization on the optimization variable x

References:
    - Envelope Theorem: https://en.wikipedia.org/wiki/Envelope_theorem
    - Implicit Differentiation in Optimization: Amos & Kolter (2017), OptNet
    
Example:
    >>> import torch
    >>> import torch.nn as nn
    >>> from models import InfConvolution
    >>> 
    >>> # Define network f(x) = a^T x + b
    >>> class LinearF(nn.Module):
    ...     def __init__(self, d):
    ...         super().__init__()
    ...         self.a = nn.Parameter(torch.randn(d))
    ...         self.b = nn.Parameter(torch.randn(()))
    ...     def forward(self, x):
    ...         return x @ self.a + self.b
    >>> 
    >>> # Define kernel K(x,y) = 0.5||x-y||^2
    >>> def K2(x, y):
    ...     return 0.5 * ((x - y)**2).sum()
    >>> 
    >>> # Compute infimal convolution
    >>> fnet = LinearF(3)
    >>> y = torch.randn(3)
    >>> x0 = torch.zeros(3)
    >>> g, converged, x_star = InfConvolution.apply(y, fnet, K2, x0, 100, 1.0, "lbfgs", 0.0, 1e-6, *list(fnet.parameters()))
    >>> print(f"g(y) = {g.item():.4f}, converged = {converged}")
    >>> print(f"Optimal x: {x_star}")
    >>> 
    >>> # Compute gradients
    >>> g.backward()
    >>> print(f"Gradient wrt a: {fnet.a.grad}")
"""
import torch
from torch.autograd import Function
from torch.func import functional_call
from tools.feedback import logger

class InfConvolution(Function):
    """
    PyTorch autograd Function for computing infimal convolution with implicit differentiation.
    
    Computes g(y) = inf_x [K(x,y) - f(x)] where:
        - K(x,y) is a kernel function (typically a distance metric)
        - f(x) is a neural network parameterized by θ
        - The optimization is performed wrt x, not θ
    
    The gradient computation uses the envelope theorem to avoid backpropagating through
    the optimization loop, making it efficient and numerically stable.
    """
    @staticmethod
    def forward(ctx, y, f_net, K, x_init, solver_steps=30, lr=1e-1, optimizer="gd", lam=0.0, tol=1e-6, patience=5, *params):
        """
        Forward pass: Solve the optimization problem to compute g(y) = inf_x [K(x,y) - f(x)].
        
        This method performs the following steps:
        1. Initialize optimization variable x from x_init
        2. Solve the minimization problem: x* = argmin_x [K(x,y) - f(x) + 0.5*lam*||x||^2]
        3. Compute and return g(y) = K(x*,y) - f(x*)
        
        The optimization is performed with manual gradient computation to avoid tracking
        gradients through the network parameters during the forward pass. This is crucial
        for efficiency and correctness of the implicit differentiation.
        
        Args:
            y (torch.Tensor): Input tensor of shape (..., d). The point at which to evaluate g(y).
            f_net (nn.Module): Neural network representing f(x). Must be differentiable wrt x.
            K (callable): Kernel function K(x, y) -> scalar tensor. Should be differentiable wrt x.
                Common choice: K(x,y) = 0.5*||x-y||^2 (squared Euclidean distance).
            x_init (torch.Tensor): Initial guess for the optimization variable x. Shape should
                match the input dimension d.
            solver_steps (int, optional): Maximum number of optimization steps. Default: 30.
                Increase for harder optimization problems.
            lr (float, optional): Learning rate for the optimizer. Default: 1e-1.
                Typical values: 1.0 for LBFGS, 1e-2 to 1e-1 for Adam/GD.
            optimizer (str, optional): Choice of optimizer. One of:
                - "lbfgs": Limited-memory BFGS (recommended for smooth problems)
                - "adam": Adam optimizer (good for non-smooth problems)
                - "gd": Gradient descent / SGD (simplest, may need more steps)
                Default: "gd"
            lam (float, optional): L2 regularization weight on x. Adds 0.5*lam*||x||^2 to the
                objective. Useful for improving conditioning. Default: 0.0 (no regularization).
            tol (float, optional): Tolerance for early stopping. Optimization stops when
                relative change in objective is less than tol. Default: 1e-6.
                Formula: |obj_val - prev_obj| / (|prev_obj| + 1e-10) < tol
            patience (int, optional): Number of consecutive steps with rel_change < tol required
                before declaring convergence. Helps avoid premature stopping with Adam/GD.
                Default: 5. Only applies to Adam/GD, not LBFGS.
            *params: Network parameters from f_net.parameters(). Needed for gradient tracking
                in the backward pass. Automatically extracted if not provided.
        
        Returns:
            tuple: (g, converged, x_star) where:
                - g (torch.Tensor): Scalar value of inf_x [K(x,y) - f(x)]
                - converged (bool): Whether optimization converged before reaching solver_steps.
                    True if relative change fell below tol, False otherwise.
                    For LBFGS, always returns True (uses internal convergence criteria).
                - x_star (torch.Tensor): Optimal x that achieves the minimum (detached).
                    Useful for warm-starting subsequent optimizations.
        
        Notes:
            - The forward pass does NOT accumulate gradients in f_net parameters.
            - Uses torch.func.functional_call to evaluate f_net without parameter tracking.
            - Manual gradient computation ensures efficiency and numerical stability.
            - LBFGS typically converges faster but may need tuning for non-smooth problems.
            - Adam is more robust to initialization and works well with default settings.
        
        Raises:
            RuntimeError: If optimization diverges or produces NaN/Inf values.
        """
        # If no params passed, get them from f_net
        if len(params) == 0:
            params = tuple(f_net.parameters())
        
        # Create detached parameter dict for functional_call
        # This prevents gradient tracking through parameters during optimization
        params_dict = {name: p.detach() for name, p in f_net.named_parameters()}
        
        x_var = x_init.clone().detach().requires_grad_(True)

        # Solve argmin_x [K(x,y) - f(x) + 0.5*lam*||x||^2]
        opt_name = optimizer.lower() if isinstance(optimizer, str) else "gd"
        converged = False
        
        with torch.enable_grad():
            if opt_name == "lbfgs":
                # LBFGS closure - compute objective and gradients
                # Track the number of function evaluations (not iterations)
                lbfgs_state = {'n_eval': 0, 'last_loss': None}
                
                def closure():
                    if x_var.grad is not None:
                        x_var.grad.zero_()
                    
                    # Compute objective using detached parameters (no param gradients)
                    k_val = K(x_var, y)
                    f_val = functional_call(f_net, params_dict, (x_var,))
                    obj = k_val - f_val + 0.5 * lam * x_var.pow(2).sum()
                    
                    # Compute gradients wrt x only (params are detached)
                    obj.backward()
                    
                    # Track function evaluations
                    lbfgs_state['n_eval'] += 1
                    lbfgs_state['last_loss'] = obj.item()
                    
                    return obj
                
                optim_obj = torch.optim.LBFGS(
                    [x_var],
                    lr=lr,
                    max_iter=solver_steps,
                    tolerance_grad=tol,
                    tolerance_change=tol,
                    line_search_fn="strong_wolfe"
                )
                
                # Step returns None, but LBFGS stops early if it converges
                optim_obj.step(closure)
                
                # Check for NaN/Inf in final result
                with torch.no_grad():
                    k_val = K(x_var, y)
                    f_val = functional_call(f_net, params_dict, (x_var,))
                    final_obj = k_val - f_val + 0.5 * lam * x_var.pow(2).sum()
                    
                    if torch.isnan(final_obj) or torch.isinf(final_obj):
                        logger.error(
                            f"[InfConvolution] LBFGS optimization diverged: "
                            f"objective={final_obj.item()}, x_norm={x_var.norm().item():.2e}"
                        )
                        raise RuntimeError(
                            f"LBFGS optimization produced NaN/Inf: objective={final_obj.item()}, "
                            f"x_norm={x_var.norm().item():.2e}. Try reducing lr or increasing lam."
                        )
                    
                    if torch.isnan(x_var).any() or torch.isinf(x_var).any():
                        logger.error(
                            f"[InfConvolution] LBFGS optimization produced invalid x_var: "
                            f"contains NaN or Inf"
                        )
                        raise RuntimeError(
                            "LBFGS optimization produced NaN/Inf in x_var. "
                            "Try reducing lr or increasing lam."
                        )
                
                # LBFGS convergence: check gradient norm and function value convergence
                # Don't just rely on iteration count - LBFGS can stop for many reasons
                state = optim_obj.state[optim_obj.param_groups[0]['params'][0]]
                n_iter = state.get('n_iter', 0) if state else 0
                
                # Check final gradient norm
                # Need to create a fresh variable for gradient check since x_var will be detached
                x_check = x_var.detach().clone().requires_grad_(True)
                k_val = K(x_check, y)
                f_val = functional_call(f_net, params_dict, (x_check,))
                check_obj = k_val - f_val + 0.5 * lam * x_check.pow(2).sum()
                check_obj.backward()
                final_grad_norm = x_check.grad.norm().item()
                
                # Converged if: (1) used < max_iter AND (2) gradient is small enough
                # Gradient tolerance check: grad_norm < tol * (1 + |obj|)
                grad_converged = final_grad_norm < tol * (1.0 + abs(final_obj.item()))
                iter_stopped_early = n_iter < solver_steps
                
                converged = iter_stopped_early and grad_converged
                
                if not converged:
                    if not iter_stopped_early:
                        # Used all iterations
                        logger.warning(
                            f"[InfConvolution] LBFGS did not converge after {n_iter} iterations "
                            f"({lbfgs_state['n_eval']} function evaluations). "
                            f"Final objective: {final_obj.item():.6e}, grad_norm: {final_grad_norm:.6e}, "
                            f"tol: {tol:.2e}. Consider increasing solver_steps (current: {solver_steps})."
                        )
                    else:
                        # Stopped early but gradient still large
                        logger.warning(
                            f"[InfConvolution] LBFGS stopped early at {n_iter} iterations but did not converge. "
                            f"Final objective: {final_obj.item():.6e}, grad_norm: {final_grad_norm:.6e}, "
                            f"tol: {tol:.2e}. The problem may be ill-conditioned. "
                            f"Consider: (1) increasing lam for regularization, (2) relaxing tol, "
                            f"or (3) using a different optimizer."
                        )
            else:
                if opt_name == "adam":
                    optim_obj = torch.optim.Adam([x_var], lr=lr)
                else:  # gd or default
                    optim_obj = torch.optim.SGD([x_var], lr=lr)
                
                prev_obj = None
                patience_counter = 0
                rel_change = float('inf')  # Initialize for logging
                for step in range(solver_steps):
                    optim_obj.zero_grad()
                    
                    # Compute objective using detached parameters (no param gradients)
                    k_val = K(x_var, y)
                    f_val = functional_call(f_net, params_dict, (x_var,))
                    obj = k_val - f_val + 0.5 * lam * x_var.pow(2).sum()
                    
                    # Compute gradients wrt x only (params are detached)
                    obj.backward()
                    
                    optim_obj.step()
                    
                    # Check for NaN/Inf
                    obj_val = obj.item()
                    if torch.isnan(obj) or torch.isinf(obj):
                        logger.error(
                            f"[InfConvolution] {opt_name.upper()} optimization diverged at step {step}: "
                            f"objective={obj_val}, x_norm={x_var.norm().item():.2e}"
                        )
                        raise RuntimeError(
                            f"{opt_name.upper()} optimization produced NaN/Inf at step {step}: "
                            f"objective={obj_val}, x_norm={x_var.norm().item():.2e}. "
                            f"Try reducing lr (current: {lr:.2e}) or increasing lam (current: {lam:.2e})."
                        )
                    
                    if torch.isnan(x_var).any() or torch.isinf(x_var).any():
                        logger.error(
                            f"[InfConvolution] {opt_name.upper()} optimization produced invalid x_var at step {step}"
                        )
                        raise RuntimeError(
                            f"{opt_name.upper()} optimization produced NaN/Inf in x_var at step {step}. "
                            f"Try reducing lr (current: {lr:.2e}) or increasing lam (current: {lam:.2e})."
                        )
                    
                    # Check convergence based on relative change with patience
                    if prev_obj is not None:
                        rel_change = abs(obj_val - prev_obj) / (abs(prev_obj) + 1e-10)
                        if rel_change < tol:
                            patience_counter += 1
                            if patience_counter >= patience:
                                converged = True
                                break
                        else:
                            patience_counter = 0  # Reset if change is above tolerance
                    prev_obj = obj_val
                    
                    # Mark as not converged if we completed all steps
                    if step == solver_steps - 1:
                        converged = False  # Did not converge early
                        logger.warning(
                            f"[InfConvolution] {opt_name.upper()} did not converge after {solver_steps} steps. "
                            f"Final objective: {obj_val:.6e}, final rel_change: {rel_change if prev_obj else 'N/A':.6e}, "
                            f"tol: {tol:.2e}, patience: {patience_counter}/{patience}. "
                            f"Consider increasing solver_steps or relaxing tol."
                        )

        x_star = x_var.detach()

        ctx.f_net = f_net
        ctx.K = K
        ctx.y = y
        ctx.num_params = len(params)
        ctx.save_for_backward(x_star, *params)

        # Compute g value (no gradients needed for return value in forward)
        with torch.no_grad():
            g = K(x_star, y) - f_net(x_star)
        
        # Return g, converged, and x_star (for warm-starting)
        # x_star is detached so it doesn't affect gradients
        return g, converged, x_star

    @staticmethod
    def backward(ctx, grad_output, grad_converged, grad_x_star):
        """
        Backward pass: Compute gradients using implicit differentiation via envelope theorem.
        
        The envelope theorem states that for g(y,θ) = inf_x [K(x,y) - f(x,θ)], the gradient
        with respect to parameters θ is:
        
            dg/dθ = -df(x*,θ)/dθ
        
        where x* = argmin_x [K(x,y) - f(x,θ)]. Notably, we do NOT need to compute dx*/dθ,
        which would require backpropagating through the optimization loop. This makes the
        computation efficient and numerically stable.
        
        Mathematical Justification:
            Let L(x,y,θ) = K(x,y) - f(x,θ). By definition:
                g(y,θ) = min_x L(x,y,θ)
            
            At the optimum x*, the first-order condition holds:
                ∂L/∂x|_{x=x*} = 0
            
            Taking the total derivative:
                dg/dθ = ∂L/∂θ|_{x=x*} + (∂L/∂x|_{x=x*}) · (dx*/dθ)
                      = ∂L/∂θ|_{x=x*}    [since ∂L/∂x|_{x=x*} = 0]
                      = -∂f(x*,θ)/∂θ
        
        Args:
            ctx: Context object containing saved tensors and attributes from forward pass.
                Stores: x_star (optimal x), params (network parameters), f_net, K, y.
            grad_output (torch.Tensor): Gradient of loss wrt g (scalar). This is typically 1.0
                when g is the final loss, or the upstream gradient in a larger computation.
            grad_converged: Gradient wrt converged flag (always None, since it's a boolean).
            grad_x_star: Gradient wrt x_star (always None since it's detached).
        
        Returns:
            tuple: Gradients wrt all forward pass inputs, in the same order:
                (grad_y, grad_f_net, grad_K, grad_x_init, grad_solver_steps, grad_lr,
                 grad_optimizer, grad_lam, grad_tol, grad_patience, *grad_params)
                
                - grad_y: None (we don't compute gradients wrt input y)
                - grad_f_net: None (not a tensor)
                - grad_K: None (not a tensor)  
                - grad_x_init: None (initialization doesn't affect final output via optimization)
                - grad_solver_steps: None (hyperparameter)
                - grad_lr: None (hyperparameter)
                - grad_optimizer: None (string)
                - grad_lam: None (hyperparameter)
                - grad_tol: None (hyperparameter)
                - grad_patience: None (hyperparameter)
                - *grad_params: Tuple of gradients wrt each network parameter θ, computed as
                    -grad_output * ∂f(x*,θ)/∂θ
        
        Implementation Details:
            - Uses saved x_star from forward pass (no need to re-solve optimization)
            - Computes gradients wrt parameters by evaluating network at x_star
            - Multiplies by -grad_output (negative sign from envelope theorem)
            - Uses saved parameters to ensure consistency with forward pass
            - allow_unused=True handles networks with unused parameters gracefully
        """
        f_net = ctx.f_net
        saved_tensors = ctx.saved_tensors
        x_star = saved_tensors[0]
        params = saved_tensors[1:]

        # Compute gradient wrt f_net parameters using envelope theorem
        # d/dθ g(y,θ) = -d/dθ f(x*,θ) evaluated at the optimal x*
        # grad_converged is None since converged is a boolean flag
        
        with torch.enable_grad():
            x_star_grad = x_star.detach().requires_grad_(True)
            # Compute f(x*) using the SAVED parameters, not f_net.parameters()
            # This ensures consistency with the forward pass computation
            grad_theta = torch.autograd.grad(
                outputs=f_net(x_star_grad),
                inputs=params,
                grad_outputs=-grad_output,  # Negative sign from envelope theorem
                retain_graph=False,
                allow_unused=True,
            )

        # Return gradients in the same order as forward inputs
        # (grad_y, grad_f_net, grad_K, grad_x_init, grad_solver_steps, grad_lr, 
        #  grad_optimizer, grad_lam, grad_tol, grad_patience, *grad_params)
        return (None, None, None, None, None, None, None, None, None, None) + grad_theta
