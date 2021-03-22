#!/usr/bin/env python3

DEFAULT_STYLE = {'plate_radius': 0.035, 'plate_length': '0.025', 'plate0_color': 'Black', 'plate_color': 'Grey', 'camera_color': 'Black'}

class URDFPrinter():
  header = """
<?xml version="1.0" ?>

<robot name="%(name)s">

  <!-- Colors -->
  <material name="Black">
    <color rgba="0 0 0 1.0"/>
  </material>
  <material name="Grey">
    <color rgba="0.2 0.2 0.2 1.0"/>
  </material>
  <material name="Orange">
    <color rgba="1.0 0.423529411765 0.0392156862745 1.0"/>
  </material>
  <material name="Blue">
  <color rgba="0.0 0.0 1 1.0"/>      
  </material>
  <material name="Red">
    <color rgba="1 0 0 1.0"/>      
  </material>
  <material name="Transparent">
    <color rgba="0 0 0 0.0"/>      
  </material>"""

  manipulator_base_template = """
  <!-- Base Link -->
  <link name="%(base_name)s">
    <inertial>
      <origin rpy="0 0 0" xyz="0 0 0"/>
      <mass value="100"/>
      <inertia ixx="9.3" ixy="0" ixz="0" iyy="9.3" iyz="0" izz="9.3"/>
    </inertial>
    <visual>
      <origin rpy="0 0 0" xyz="0 0 0"/>
      <geometry>
        <box size="%(base_block_size)s"/>
      </geometry>
      <material name="Black"/>
    </visual>
  </link>"""

  section_template = """
  <!-- Section %(index)d -->

  <joint name="Joint_%(index)da" type="revolute">
    <parent link="%(parent)s"/>
    <child link="Block%(index)d"/>
    <origin rpy="0 0 0" xyz="0 0 %(joint_z)f"/>
    <axis xyz="0 1 0"/>
    <limit effort="1000.0" lower="-1.0" upper="1.0" velocity="1000.0"/>
  </joint>

  <link name="Block%(index)d">
    <inertial>
      <origin rpy="0 0 0" xyz="0 0 0"/>
      <mass value="1"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
    <visual>
      <origin rpy="0 0 0" xyz="0 0 0"/>
      <geometry>
        <box size="%(block_size)s"/>
      </geometry>
      <material name="%(block_color)s"/>
    </visual>
  </link>

  <joint name="Joint_%(index)db" type="revolute">
    <parent link="Block%(index)d"/>
    <child link="PlateFixLink%(index)d"/>
    <origin rpy="0 0 0" xyz="0 0 0"/>
    <axis xyz="-1 0 0"/>
    <limit effort="1000.0" lower="-1.0" upper="1.0" velocity="1000.0"/>
  </joint>

  <link name="PlateFixLink%(index)d">
    <inertial>
      <origin rpy="0 0 0" xyz="0 0 0"/>
      <mass value="1"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
    <visual>
      <origin rpy="0 0 0" xyz="0 0 0"/>
      <geometry>
        <box size="%(block_size)s"/>
      </geometry>
      <material name="%(block_color)s"/>
    </visual>
  </link>

  <joint name="PlateFixJoint%(index)db" type="fixed">
    <parent link="PlateFixLink%(index)d"/>
    <child link="Plate%(index)d"/>
    <origin rpy="0 0 0" xyz="0 0 %(joint_z)f"/>
    <axis xyz="-1 0 0"/>
  </joint>

  <link name="Plate%(index)d">
    <inertial>
      <origin rpy="0 0 0" xyz="0 0 0"/>
      <mass value="1"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
    <visual>
      <origin rpy="0 0 0" xyz="0 0 0"/>
      <geometry>
        <cylinder radius="%(plate_radius)f" length="%(plate_length)f"/>
      </geometry>
      <material name="%(plate_color)s"/>
    </visual>
  </link>"""

  footer = """
</robot>"""

  manipulator_camera_template = """
  <joint name="Joint_Camera" type="fixed">
    <parent link="%(parent_link)s"/>
    <child link="CameraBox"/>
    <origin rpy="0 0 0" xyz="0 0 %(camera_z)f"/>
    <axis xyz="0 0 1"/>
  </joint>

  <link name="CameraBox">
    <inertial>
      <origin rpy="0 0 0" xyz="0 0 0"/>
      <mass value="1"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
    <visual>
      <origin rpy="0 0 0" xyz="0 0 0"/>
      <geometry>
        <sphere radius="%(size)f"/>
      </geometry>
      <material name="%(color)s"/>
    </visual>
  </link>"""

  target_template = """
<link name="base_link">
  <origin rpy="0 0 0" xyz="%(xyz)s"/>
  <inertial>
    <origin rpy="0 0 0" xyz="0 0 0"/>
    <mass value="1"/>
    <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
  </inertial>
  <visual>
    <geometry>
      <sphere radius="%(r)f"/>
    </geometry>
    <material name="Red"/>
  </visual>
</link>
"""

  base_name = "SprutBase"

  def print_manipulator(self, f, NS, NP, style=DEFAULT_STYLE):
    self.print_header(f, "Manipulator")
    last_plate = self.print_manipulator_body(f, NS, NP, style)
    self.print_manipulator_camera(f, last_plate, style)
    self.print_footer(f)

  def print_cage(self, f):
    self.print_header(f, "Cage")
    self.print_cage_body(f)
    self.print_footer(f)

  def print_target(self, f):
    self.print_header(f, "Target")
    self.print_target_body(f)
    self.print_footer(f)

  def print_header(self, f, name):
    print(self.header % {"name": name}, file=f)

  def print_footer(self, f):
    print(self.footer,file = f)
    
  def print_manipulator_body(self, f, NS, NP, style):
    """
    Prints all segments of the manipulator's body
    Returns the name of the very last link (to mount the camera on)
    """
    print(self.manipulator_base_template % {"base_name": self.base_name, 'base_block_size': '0.05 0.05 0.005'}, file=f)

    JD = 0.028 # 4mm plate + 24mm to the axis
    NJ = NS * NP
    for i in range(NJ):
      if i > 0:
        parent = "Plate%d" % (i-1)
      else:
        parent = self.base_name

      if (i + 1) % NP == 0:
        plate_color = style['plate0_color']
      else:
        plate_color = style['plate_color']

      print(self.section_template % {
        'parent': parent, 'index': i,
        'block_size': '0.023 0.023 0.005',
        'block_color': style['block_color'],
        'plate_color': plate_color,
        'plate_radius': style['plate_radius'],
        'plate_length': style['plate_length'],
        'joint_z': JD}, file=f)

    return "Plate%d" % (NJ-1)

  def print_manipulator_camera(self, f, parent_link, style):
    print(self.manipulator_camera_template % {
      "parent_link": parent_link,
      "size": 0.005,
      "camera_z": 0, #0.035,
      "color": style['camera_color']
      }, file=f)

  def print_cage_body(self, f):
    print("<!-- fixme cage -->", file=f)

  def print_target_body(self, f):
    print(self.target_template % {"xyz": "1 1 1", "r": 0.05}, file=f)
