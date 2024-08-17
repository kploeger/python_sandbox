import math

#mesh_name = "basket_ring_convex_part_side"
#geom_class = "basket_ring"

n_pieces = 36
sub_angle = 2. * math.pi / n_pieces

for i_piece in range(n_pieces):
    euler_z = sub_angle * i_piece
    print(f'''
<collision>
    <origin xyz="0 0 0" rpy="0 0 {euler_z}"/>
    <geometry>
        <mesh filename="${{mesh_base_dir}}/cone_convex_part_side.obj"/>
    </geometry>
</collision>''')
