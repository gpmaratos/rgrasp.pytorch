from torch.nn.functional import smooth_l1_loss, binary_cross_entropy

class Loss:
    """
    Loss metric. Calculates the final loss.
    """
    def __init__(self, cfg):
        b_factor = cfg.b_factor
        alpha = cfg.alpha
        self.b_factor = b_factor
        self.alpha = alpha

    def __call__(self, p_reg, p_cls, t_reg, t_cls, P):
        reg_loss = smooth_l1_loss(p_reg, t_reg)
        cls_loss = binary_cross_entropy(p_cls, t_cls)
        final_loss = (cls_loss + self.alpha*reg_loss)/\
                        (P + P*self.b_factor)
        return final_loss
