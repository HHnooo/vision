from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package="camera_data_handle",
            executable='receive_data',
            name='receive_data',
            #output="screen"
            parameters=[
                './src/camera_data_handle/config/param.yaml']
        )
        # ,
        # Node(
        #     package="camera_data_handle",
        #     executable='tf_publish',
        #     name='tf_publish'
        # )
    ])

# from launch import LaunchDescription
# from launch_ros.actions import Node
# from launch.actions import ExecuteProcess
# from launch.substitutions import FindExecutable
# from ament_index_python.packages import get_package_share_directory
# import os
#
# def generate_launch_description():
#     rviz_config_path = os.path.join(
#         get_package_share_directory('your_package_name'),
#         'config',
#         'camera_tf.rviz'
#     )
#
#     if not os.path.exists(rviz_config_path):
#         rviz_config_path = ''
#         print("Warning: RViz config file not found, using default configuration")
#
#     return LaunchDescription([
#         ExecuteProcess(
#             cmd=[
#                 'xterm', '-e',
#                 FindExecutable(name='ros2'), 'run', 'your_package_name', 'camera_node'
#             ],
#             output='screen',
#             name='camera_node_terminal',
#             description='Starts camera node in a new terminal'
#         ),
#
#         ExecuteProcess(
#             cmd=[
#                 'xterm', '-e',
#                 FindExecutable(name='ros2'), 'run', 'your_package_name', 'tf_transform_node'
#             ],
#             output='screen',
#             name='tf_node_terminal',
#             description='Starts TF transform node in a new terminal'
#         ),
#
#         ExecuteProcess(
#             cmd=[
#                 'xterm', '-e',
#                 FindExecutable(name='rviz2'),
#                 '-d', rviz_config_path
#             ] if rviz_config_path else [
#                 'xterm', '-e',
#                 FindExecutable(name='rviz2')
#             ],
#             output='screen',
#             name='rviz_terminal',
#             description='Starts RViz2 in a new terminal for TF visualization'
#         )
#     ])
