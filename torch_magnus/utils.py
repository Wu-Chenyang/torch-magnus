import torch

@torch.compile
def arnoldi_iteration(
    A_dense: torch.Tensor,
    v_batch: torch.Tensor,
    k: int,
    re_orth: bool = False,
):
    r"""
    Batched (block-independent) Arnoldi iteration producing a Krylov basis **Q**
    and the small upper-Hessenberg matrix **H**.

    Parameters
    ----------
    A_dense : torch.Tensor, shape (m, m) **or** (B, m, m)
        System matrix.  
        * If a single (m, m) matrix is supplied it is broadcast to all B
          batch items.  
        * If each batch owns its own matrix pass a full (B, m, m) tensor.

    v_batch : torch.Tensor, shape (B, m)
        Batch of B starting / seed vectors.

    k : int
        Dimension of the Krylov subspace to be generated (i.e. we build
        `q₀, …, q_k` → **H** is (k+1)×k).

    re_orth : bool, default = False
        Apply the “double re-orthogonalisation’’ recommended by Saad to
        improve numerical stability when k is large or A is highly non-normal.

    Returns
    -------
    Q : torch.Tensor, shape (B, m, k+1)
        Orthonormal basis vectors stored column-wise.

    H : torch.Tensor, shape (B, k+1, k)
        Projected upper-Hessenberg matrix satisfying  
        `A Q_k  =  Q_{k+1} H_k`.

    beta : torch.Tensor, shape (B,)
        Norms of the original seed vectors ‖v_b‖.  Needed for
        expm-multiply reconstruction.

    """
    # --- Implementation identical to the user's original -------------------
    B, m = v_batch.shape
    if A_dense.dim() == 2:
        A_dense = A_dense.expand(B, -1, -1)

    Q = torch.zeros(B, m, k + 1, dtype=v_batch.dtype, device=v_batch.device)
    H = torch.zeros(B, k + 1, k, dtype=v_batch.dtype, device=v_batch.device)

    # normalise q₀
    beta = torch.linalg.norm(v_batch, dim=1, keepdim=True)
    Q[:, :, 0] = v_batch / beta

    for j in range(k):
        w = torch.bmm(A_dense, Q[:, :, j:j + 1]).squeeze(-1)

        # classical Gram–Schmidt
        for i in range(j + 1):
            H[:, i, j] = torch.sum(w * Q[:, :, i], dim=1)
            w = w - H[:, i, j:j + 1] * Q[:, :, i]

        if re_orth:                       # optional double pass
            for _ in range(2):
                for i in range(j + 1):
                    corr = (w * Q[:, :, i]).sum(dim=1)
                    w = w - corr[:, None] * Q[:, :, i]

        if j < k - 1:
            H[:, j + 1, j] = torch.linalg.norm(w, dim=1)
            mask = H[:, j + 1, j] > 1e-12
            if mask.any():
                Q[mask, :, j + 1] = w[mask] / H[mask, j + 1, j:j + 1]

    return Q, H, beta.squeeze(1)


@torch.compile
def expm_multiply(
    A_dense: torch.Tensor,
    v_batch: torch.Tensor,
    t_batch: torch.Tensor | float,
    k: int = 30,
    tol: float = 1e-5,
):
    r"""
    Compute a **batched action of the matrix exponential**
    :math:`w_b \approx \exp(t_b A) v_b` via the Arnoldi / Krylov method.

    Strategy
    --------
    1. Build a (k+1)-step Arnoldi factorisation  
       :math:`A Q_k = Q_{k+1} H_k`.
    2. Replace the large exponential by the small one  
       :math:`\exp(t_b A) v_b ≈ β_b Q_k \exp(t_b H_k) e₁`.
    3. Use Saad’s a-posteriori error estimator  
       :math:`err_b ≈ |t_b h_{k+1,k} (e_k^T \exp(t_b H_k) e_1)|`
       to stop early whenever all batches are below **tol**.

    Parameters
    ----------
    A_dense : torch.Tensor, shape (m, m) **or** (B, m, m)
        System matrix (broadcast rules identical to :func:`arnoldi_iteration`).

    v_batch : torch.Tensor, shape (B, m)
        B right-hand-side vectors.

    t_batch : float or torch.Tensor, shape () **or** (B,)
        Time/scale parameters.  A scalar is broadcast to every batch.

    k : int, default = 30
        Maximum Arnoldi subspace dimension.

    tol : float, default = 1e-5
        Relative residual tolerance for the Saad estimator.

    Returns
    -------
    w_batch : torch.Tensor, shape (B, m)
        Batched approximation of :math:`\exp(t_b A) v_b`.

    When is this method **most effective**?
    ---------------------------------------
    * **Large, sparse, or “mat-vec cheap’’ operators** – memory never stores
      `exp(A)`; only `A·q` is required.
    * **Normal or nearly-normal spectra that live in the *left* half-plane**
      (Re λ ≪ 0).  Exponential decay lets small-k Krylov capture the action
      quickly (often k ≲ 30 for 1e-6 accuracy).
    * **Symmetric/Hermitian matrices** – you may switch to Lanczos for half
      the memory, but the present code still converges very fast.
    * Mildly non-normal systems such as many dissipative PDE discretisations.

    Limitations
    -----------
    * **Highly non-normal or Jordan-rich matrices** – k may need to be very
      large or require restarted schemes.  Pseudospectral growth destroys
      the simple Saad bound.
    * **Purely imaginary spectra** (skew-Hermitian quantum Hamiltonians):
      `|exp(iθ)| = 1`, thus no decay ⇒ polynomial approximation lengthens
      (k ≈ 100+ is common).
    * For *tiny dense* matrices (m ≲ 200) a direct
      `torch.linalg.matrix_exp(A) @ v` is usually faster.

    """
    # ------------ Arnoldi stage -------------------------------------------
    Q, H, beta = arnoldi_iteration(A_dense, v_batch, k)  # Q:(B,m,k+1), H:(B,k+1,k)

    B, _, _ = Q.shape
    if not torch.is_tensor(t_batch):
        t_batch = torch.tensor([t_batch] * B, dtype=Q.dtype, device=Q.device)
    t_batch = t_batch.to(Q.device).reshape(B, 1, 1)

    # ------------ Adaptive truncation -------------------------------------
    for k_used in range(1, k + 1):
        Hk = H[:, :k_used, :k_used]
        F = torch.linalg.matrix_exp(t_batch * Hk)

        if k_used < k:
            err = torch.abs(
                t_batch.squeeze() * H[:, k_used, k_used - 1] * F[:, k_used - 1, 0]
            )
            if torch.all(err < tol):
                break

    # ------------ Reconstruction: w = β Q_k F[:,0] ------------------------
    F0 = F[:, :, 0]                                   # (B, k_used)
    Qk = Q[:, :, :k_used]                             # (B, m, k_used)
    w_batch = beta[:, None] * torch.bmm(Qk, F0.unsqueeze(-1)).squeeze(-1)
    return w_batch
