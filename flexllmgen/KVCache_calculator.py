import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import matplotlib.patches as patches
import argparse
import json
import matplotlib.ticker as ticker

# 自定义格式化函数
def format_k(x_val, pos):
    if x_val == 1000:
        return '1K'
    elif x_val == 10000:
        return '10K'
    elif x_val == 100000:
        return '100K'
    elif x_val == 1000000:
        return '1M'
    else:
        return ''

def draw_memory_cost(args):
    # 绘图参数全家桶
    params = {
        'axes.labelsize': '13',
        'xtick.labelsize': '11',
        'ytick.labelsize': '10',
        'legend.fontsize': '10',
        'figure.figsize': '3.5, 2.5',
        'figure.dpi':'300',
        'figure.subplot.left':'0.15',
        'figure.subplot.right':'0.96',
        'figure.subplot.bottom':'0.14',
        'figure.subplot.top':'0.91',
        'pdf.fonttype':'42',
        'ps.fonttype':'42',
    }
    pylab.rcParams.update(params)
    # 设置字体
    #plt.rcParams['font.family'] = 'serif'
    #plt.rcParams['font.serif'] = ['Times New Roman']

    color_1 = "#F27970"
    color_2 = "#BB9727"
    color_3 = "#54B345"
    color_4 = "#32B897"
    color_5 = "#05B9E2"
    # 创建图表和轴
    fig, ax = plt.subplots(figsize=(8, 4))

    # 定义数据
    x_max_value = args.context_length
    model = args.model_name
    element_size = args.element_size
    model_weights_height = args.model_size * element_size
    num_layers = args.num_layers
    num_kv_head = args.num_kv_head
    head_dim = args.head_dim
    device_name = args.device_name
    gpu_memory_limit = args.gpu_memory_limit
    batch_size = args.batch_size
    cpu_memory_limit = args.cpu_memory_limit


    # draw weight
    token_ids = np.arange(0, x_max_value)
    ax.fill_between(token_ids, 0 , model_weights_height, color=color_2, alpha=0.7, label='Model weights')

    # draw kv cache
    kv_size_each_token = (2*num_layers*num_kv_head*head_dim*element_size)/1000000000
    kv_size = kv_size_each_token * x_max_value
    kv_size = kv_size * batch_size
    print(kv_size)
    # 创建多边形
    vertices = []
    vertices.append((0, model_weights_height))
    vertices.append((x_max_value, model_weights_height ))
    vertices.append((x_max_value, model_weights_height + kv_size))
    kv_polygon = patches.Polygon(vertices, closed=True,
                               facecolor=color_4, alpha=0.7, label='KV cache')
    ax.add_patch(kv_polygon)


    # 添加"Memory Overflow"文本标签
    ax.axhline(y=gpu_memory_limit, color='red', linestyle='--', linewidth=2)
    ax.text(0, gpu_memory_limit+4, 'HBM', color='red', fontsize=12,
           fontweight='bold', ha='left', va='baseline',
           )
    ax.axhline(y= cpu_memory_limit + gpu_memory_limit, color='red', linestyle='--', linewidth=2)
    ax.text(0, cpu_memory_limit + gpu_memory_limit+4, 'DRAM', color='red', fontsize=12,
           fontweight='bold', ha='left', va='baseline',
           )

    # 设置轴标签和标题
    ax.set_xlabel('Context Length')
    ax.set_ylabel('Memory Cost (GB)')
    ax.set_xlim(0, x_max_value)
    ax.set_ylim(0, 500)
    plt.title(f"{model}, BS={batch_size}, {device_name}")  # 添加标题
    # modify x-axis
    ax.set_xlim(0, x_max_value)

    # 设置 x 轴只在这些位置显示刻度
    if args.custom_xticks:
        custom_xticks = [10000, 100000, 1000000]
        ax.set_xticks(custom_xticks)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_k))

    # 添加图例
    ax.legend(loc='upper center', ncol=3)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.grid(linestyle="--", linewidth=0.5, color='black', alpha = 0.3)
    # 保存为PDF文件

    plt.savefig(f'{model}_BS_{batch_size}_memory_cost_{device_name}.pdf', format='pdf', bbox_inches='tight', dpi=300)
    # 如果你想同时显示图表，保留这行
    plt.show()



def add_parser_args(parser):
    parser.add_argument('--device_name', type=str, default='RTX3090', help="device name")
    parser.add_argument('--gpu_memory_limit', type=int, default=24, help="GPU memory limit")
    parser.add_argument('--cpu_memory_limit', type=int, default=32, help="CPU memory limit")
    parser.add_argument('--context_length', type=int, default=1280000, help="context length")
    parser.add_argument('--batch_size', type=int, default=1, help="batch size")
    parser.add_argument('--model_size', type=float, default=32, help="model size")
    parser.add_argument("--model_config", type=str, default=None, help="Path to model config file")
    parser.add_argument('--model_name', type=str, default='Qwen_8B', help="model name")
    parser.add_argument('--num_layers', type=int, default=36, help="number of layers")
    parser.add_argument('--num_kv_head', type=int, default=8, help="kv cache size")
    parser.add_argument('--head_dim', type=int, default=128, help="head dimension")
    parser.add_argument('--element_size', type=int, default=2, help="fp16=2byte")
    parser.add_argument('--custom_xticks', type=bool, default=False, help="custom xticks")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='KV Cache Calculator')
    add_parser_args(parser)
    args = parser.parse_args()

    with open("./config/device_config.json", 'r', encoding='utf-8') as f:
        device_config = json.load(f)
        if args.device_name in device_config:
           args.gpu_memory_limit = device_config[args.device_name]
           args.cpu_memory_limit = device_config["CPU"]
        else:
            raise Exception("Unknown device name")

    if args.model_config is not None:
        with open(args.model_config, 'r', encoding='utf-8') as f:
            config = json.load(f)
        args.model_name = config["model_type"]
        args.num_layers = config["num_hidden_layers"]
        if "head_dim" not in config:
            args.head_dim = config["hidden_size"] / config["num_attention_heads"]
        args.num_kv_head = config["num_key_value_heads"]

    draw_memory_cost(args)