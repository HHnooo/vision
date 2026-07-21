from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'video_path',
            default_value='../vidios/vidio0001.mp4',
            description='视频文件路径'
        ),
        DeclareLaunchArgument(
            'loop',
            default_value='true',
            description='是否循环播放'
        ),
        Node(
            package="camera_data_handle",
            executable='camera_debug',
            name='camera_debug',
            output="screen",
            parameters=[{
                'video_path': LaunchConfiguration('video_path'),
                'loop': LaunchConfiguration('loop'),
            }]
        )
    ])
