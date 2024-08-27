# 简介

使用中的系统盘，随后找到操作中的`扩容`，将其扩充至200~400G即可。
首先，我们需要前往阿里云进入服务器界面，我们需要依次选择云服务器ECS——立即购买——按量付费；
随后，你需要完成以下操作：
1. 在第3页找到通用型g8i `ecs.g8i.6xlarge  24vCPU` 对应型号，镜像选择 Ubuntu 22.04 64位
2. 找到带宽和安全组，点选公网IP中的 `分配公网IPv4地址`
3. 选择`自定义密码`并设置密码后即可确认下单（价格约一小时6.321元）

至此，服务器实例创建完毕，你也拥有了一个可以公网访问的IP地址。（记得密码要写的复杂！否则容易被攻击，建议大小写特殊符号都要有）

> **❗ 重要信息**：如果您不会持续使用相关服务器实例，可以考虑下列方案,在保存现有代码和模型等数据的同事，节省费用支出：
>
> 请在实例界面选择停止实例后选择 **节省停机模式** ，待下次进入时可正常恢复开发环境，同时节约计费；如果你想完全停止所有实例计费，你需要在`更多操作`中完全释放实例，若仍是担心费用问题，可在左侧的 **块存储（云盘）** 处检查硬盘资源是否成功释放。


注意，在创建实例后，推荐在正式进入服务器之前，先进行存储的扩容。我们可以点击左侧`存储与快照`下的`块存储（云盘）`，看到当前有个
请参加比赛的各团队伙伴根据项目实际带宽及流量需求，综合考虑后选择合适的流量/带宽计费方案，可以根据实际情况需求进行动态修改。


# 一、安装环境

创建实例后，点击远程连接即可进入机器，但此时不方便操作，我们可以通过 vscode ssh 远程连接到服务器实例，通过密码验证直接登录 `ssh root@xxx.xxx.xxx.xxx`。进入机器后，我们需要安装最新的 IPEX-LLM 程序，你也可以直接把这个notebook移动到服务器上 `/home` 目录下进行操作。vscode需要安装python和jupyter依赖等。

> 我们并不需要拘泥于当前的 IPEX-LLM 技术方案，欢迎大家使用各种比赛规则中推荐的部署方案进行项目实现。    
>
> 本notebook仅提供一种部署及实现RAG的参考方案，更多方案、以及资料请参考：   
> - [OpenVINO LLMs](https://docs.llamaindex.ai/en/stable/examples/llm/openvino/)  
> - [llm-rag-langchain-with-output](https://docs.openvino.ai/nightly/notebooks/llm-rag-langchain-with-output.html)
> - [llm-rag-llamaindex-with-output](https://docs.openvino.ai/nightly/notebooks/llm-rag-llamaindex-with-output.html)
> - [Intel® Extension for Transformers](https://github.com/intel/intel-extension-for-transformers/blob/main/docs/weightonlyquant.md#examples-for-gpu)
> - [xFasterTransformer](https://github.com/intel/xFasterTransformer)
> - [Intel Extension for Pytorch](https://github.com/intel/intel-extension-for-pytorch)
