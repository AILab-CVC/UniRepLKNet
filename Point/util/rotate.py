import torch

def rotate_angle_vector(theta, v):
    '''
        theta: B 1
        v:  B 3
    '''
    cos_a = torch.cos(theta)
    sin_a = torch.sin(theta)
    x, y, z = v[:, 0:1], v[:, 1:2], v[:, 2:3]
    
    R = torch.stack([
        torch.cat([cos_a+(1-cos_a)*x*x, (1-cos_a)*x*y-sin_a*z, (1-cos_a)*x*z+sin_a*y], dim=-1) , # [b1 b1 b1]
        torch.cat([(1-cos_a)*y*x+sin_a*z, cos_a+(1-cos_a)*y*y, (1-cos_a)*y*z-sin_a*x], dim=-1) ,
        torch.cat([(1-cos_a)*z*x-sin_a*y, (1-cos_a)*z*y+sin_a*x, cos_a+(1-cos_a)*z*z], dim=-1) 
    ], dim = 1)

    return R

def rotate_theta_phi(angles):
    '''
        angles: B, 2
    '''
    assert len(angles.shape) == 2
    B = angles.size(0)
    theta, phi = angles[:, 0:1], angles[:, 1:2]

    v1 = torch.Tensor([[0, 0, 1]]).expand(B, -1) # B 3
    v2 = torch.cat([torch.sin(theta) , -torch.cos(theta), torch.zeros_like(theta)], dim=-1) # B 3

    R1_inv = rotate_angle_vector(-theta, v1)
    R2_inv = rotate_angle_vector(-phi, v2)
    R_inv = R1_inv @ R2_inv

    return R_inv

def rotate_point_clouds(pc, rotation_matrix, use_normals=False):
    '''
        Input: 
            pc  B N 3
            R   3 3
        Output:
            B N 3
    '''
    if not use_normals:
        new_pc = torch.einsum('bnc, dc -> bnd', pc, rotation_matrix.float().to(pc.device))
    else:
        new_pc = torch.einsum('bnc, dc -> bnd', pc[:, :, :3], rotation_matrix.float().to(pc.device))
        new_normal = torch.einsum('bnc, dc -> bnd', pc[:, :, 3:], rotation_matrix.float().to(pc.device))
        new_pc = torch.cat([new_pc, new_normal], dim=-1)
    return new_pc

def rotate_point_clouds_batch(pc, rotation_matrix, use_normals=False):
    '''
        Input: 
            pc  B N 3
            R   B 3 3
        Output:
            B N 3
    '''
    if not use_normals:
        new_pc = torch.einsum('bnc, bdc -> bnd', pc, rotation_matrix.float().to(pc.device))
    else:
        new_pc = torch.einsum('bnc, bdc -> bnd', pc[:, :, :3], rotation_matrix.float().to(pc.device))
        new_normal = torch.einsum('bnc, bdc -> bnd', pc[:, :, 3:], rotation_matrix.float().to(pc.device))
        new_pc = torch.cat([new_pc, new_normal], dim=-1)
    return new_pc
