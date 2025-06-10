Name:           sysHAX
Version:        0.2.0
Release:        1
Summary:        sysHAX 动态调度服务

License:        Mulan PSL v2
URL:            https://gitee.com/openeuler/sysHAX
Source0:        %{name}-%{version}.tar.gz

BuildArch:      noarch
BuildRequires:  python3-devel
Requires:       python3, python3-fastapi >= 0.115.0, python3-uvicorn >= 0.23.2, python3-httpx >= 0.24.1, python3-PyYAML >= 6.0.1

%description
sysHAX 是 vLLM Prefill-Decode 分离调度中间层，接收客户端请求并在 GPU 实例（prefill）与 GPU/CPU 实例（decode）之间动态调度。
通过实时监控和负载决策，在高并发场景下提升系统吞吐量。

%prep
%setup -q -n sysHAX
sed -i '/import yaml/a sys.path.insert(0, "%{_datadir}/sysHAX")' cli.py
sed -i 's|sys.path.insert(0, str(Path(__file__).parent))|sys.path.insert(0, "%{_datadir}/sysHAX")|' cli.py
sed -i '/^def get_version()/,/^$/c\
def get_version() -> str:\
    return "%{version}"' cli.py

%build
# 无需编译步骤

%install
# 安装应用到共享目录
mkdir -p %{buildroot}%{_datadir}/sysHAX
cp -a %{_builddir}/%{name}/main.py %{buildroot}%{_datadir}/sysHAX/main.py
cp -a %{_builddir}/%{name}/src %{buildroot}%{_datadir}/sysHAX/
mkdir -p %{buildroot}%{_datadir}/sysHAX/config
cp -a %{_builddir}/%{name}/config/config.example.yaml %{buildroot}%{_datadir}/sysHAX/config/config.example.yaml

# 安装命令行脚本 syshax
install -Dm 0755 %{_builddir}/%{name}/cli.py %{buildroot}%{_bindir}/syshax

%files
# 许可证和文档
%license License/LICENSE
%doc README.md README.en.md

# 应用目录
%dir %attr(0755,root,root) %{_datadir}/sysHAX
%dir %attr(0755,root,root) %{_datadir}/sysHAX/src
%dir %attr(0755,root,root) %{_datadir}/sysHAX/config

# 应用文件
%attr(0644,root,root) %{_datadir}/sysHAX/main.py
%attr(0644,root,root) %{_datadir}/sysHAX/src/*
%attr(0644,root,root) %{_datadir}/sysHAX/config/config.example.yaml

# 命令行脚本
%attr(0755,root,root) %{_bindir}/syshax
%exclude %{_datadir}/sysHAX/__pycache__/*

%post
# 安装后创建默认配置文件，避免运行时缺少 config.yaml
if [ ! -f %{_datadir}/sysHAX/config/config.yaml ]; then
    cp %{_datadir}/sysHAX/config/config.example.yaml %{_datadir}/sysHAX/config/config.yaml
fi

%postun -p /bin/sh
# 卸载时删除配置文件
rm -f %{_datadir}/sysHAX/config/config.yaml

%changelog
* Wed May 21 2025 Huawei Technologies Co., Ltd. <youremail@huawei.com> - 0.1.0-1
- Initial package 
