# bedrock Demo

## 支持功能
### Tool Use
* 上传自己的tool到tools文件夹，即可在调用大模型时使用
* tool定义时需要：
   * 包含tool_config与invoke两个方法
   * invoke方法实际操作，tool_config声明tool的schema
   * tool_config中的tool_name需要与tool脚本名一致

## Log
11.27.2024 支持weather_tool
11.28 2024 支持file_reader file_packer_for_lambda
12.06 2024 支持computer use中的computer工具
