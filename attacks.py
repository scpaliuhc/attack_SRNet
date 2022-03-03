import tensorflow as tf
def FGSM_tf_pre(loss, x, eps, clip=None):
    """[summary]

    Parameters
    ----------
    loss : [type]
        [description]
    x : [type]
        [description]
    eps : [type]
        [description]
    clip_b : [type]
        [description]
    clip_u : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """    
    grad=tf.gradients(loss, x)[0]
    adv_x = x + eps*tf.sign(grad)
    if clip:
        adv_x = tf.clip_by_value(adv_x,clip[0],clip[1])
    return adv_x

def MI_FGSM_tf_pre(loss, x, g_t, eps, u, clip=None):
    """[summary]

    Parameters
    ----------
    loss : [type]
        [description]
    x : [type]
        [description]
    eps : [type]
        [description]
    u : [type]
        [description]
    clip : [type], optional
        [description], by default None
    """ 
    grad=tf.gradients(loss, x)[0] 
    grad_l1_norm=tf.norm(grad,ord=1) 
    adv_x = x + eps*tf.sign(u*g_t+grad/grad_l1_norm)
    if clip:
        adv_x = tf.clip_by_value(adv_x,clip[0],clip[1])
    return adv_x, grad, grad_l1_norm




