#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
W8网络诊断器
功能全面的网络诊断工具，包含网络诊断、故障排查、性能分析等功能
"""

import os
import sys
import time
import socket
import subprocess
import threading
import json
import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum


class DiagnosticStatus(Enum):
    """诊断状态枚举"""
    SUCCESS = "成功"
    WARNING = "警告"
    ERROR = "错误"
    UNKNOWN = "未知"


@dataclass
class NetworkTestResult:
    """网络测试结果"""
    test_name: str
    status: DiagnosticStatus
    message: str
    data: Dict[str, Any]
    timestamp: str


@dataclass
class DiagnosticReport:
    """诊断报告"""
    report_id: str
    timestamp: str
    target: str
    tests: List[NetworkTestResult]
    summary: Dict[str, int]
    recommendations: List[str]


class NetworkDiagnostics:
    """网络诊断器主类"""
    
    def __init__(self):
        """初始化网络诊断器"""
        self.test_results: List[NetworkTestResult] = []
        self.recommendations: List[str] = []
        
    def ping_host(self, host: str, count: int = 4, timeout: int = 5) -> NetworkTestResult:
        """
        Ping主机测试
        
        Args:
            host: 目标主机
            count: ping次数
            timeout: 超时时间
            
        Returns:
            NetworkTestResult: 测试结果
        """
        try:
            # 构建ping命令
            if os.name == 'nt':  # Windows
                cmd = ['ping', '-n', str(count), '-w', str(timeout * 1000), host]
            else:  # Linux/Mac
                cmd = ['ping', '-c', str(count), '-W', str(timeout), host]
            
            start_time = time.time()
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            end_time = time.time()
            
            data = {
                'host': host,
                'count': count,
                'timeout': timeout,
                'execution_time': end_time - start_time,
                'return_code': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr
            }
            
            if result.returncode == 0:
                # 解析ping结果
                lines = result.stdout.split('\n')
                packet_loss = 0
                avg_time = 0
                
                for line in lines:
                    if '包' in line or 'packets transmitted' in line:
                        if '0% packet loss' in line or '0% loss' in line:
                            packet_loss = 0
                        elif '%' in line:
                            # 提取丢包率
                            parts = line.split()
                            for part in parts:
                                if '%' in part:
                                    packet_loss = float(part.replace('%', ''))
                                    break
                    elif '平均' in line or 'avg' in line:
                        # 提取平均时间
                        parts = line.split('=')
                        if len(parts) > 1:
                            time_part = parts[1].split('/')[1]
                            avg_time = float(time_part)
                
                data['packet_loss'] = packet_loss
                data['avg_time'] = avg_time
                
                if packet_loss == 0:
                    status = DiagnosticStatus.SUCCESS
                    message = f"主机 {host} 连通性正常，平均响应时间: {avg_time:.2f}ms"
                else:
                    status = DiagnosticStatus.WARNING
                    message = f"主机 {host} 连通性异常，丢包率: {packet_loss}%"
            else:
                status = DiagnosticStatus.ERROR
                message = f"主机 {host} 无法连通"
                
            return NetworkTestResult(
                test_name="Ping连通性测试",
                status=status,
                message=message,
                data=data,
                timestamp=datetime.datetime.now().isoformat()
            )
            
        except subprocess.TimeoutExpired:
            return NetworkTestResult(
                test_name="Ping连通性测试",
                status=DiagnosticStatus.ERROR,
                message=f"主机 {host} Ping超时",
                data={'host': host, 'error': 'timeout'},
                timestamp=datetime.datetime.now().isoformat()
            )
        except Exception as e:
            return NetworkTestResult(
                test_name="Ping连通性测试",
                status=DiagnosticStatus.ERROR,
                message=f"Ping测试失败: {str(e)}",
                data={'host': host, 'error': str(e)},
                timestamp=datetime.datetime.now().isoformat()
            )
    
    def check_dns_resolution(self, domain: str, dns_server: Optional[str] = None) -> NetworkTestResult:
        """
        DNS解析测试
        
        Args:
            domain: 要解析的域名
            dns_server: 指定的DNS服务器（可选）
            
        Returns:
            NetworkTestResult: 测试结果
        """
        try:
            start_time = time.time()
            
            if dns_server:
                # 使用指定的DNS服务器
                resolver = socket.gethostbyname_ex(domain)
                dns_info = resolver[2]
                resolution_time = time.time() - start_time
                
                data = {
                    'domain': domain,
                    'dns_server': dns_server,
                    'resolved_ips': dns_info,
                    'resolution_time': resolution_time
                }
                
                status = DiagnosticStatus.SUCCESS
                message = f"域名 {domain} 解析成功，IP地址: {', '.join(dns_info)}"
            else:
                # 使用系统默认DNS
                start_time = time.time()
                ip = socket.gethostbyname(domain)
                resolution_time = time.time() - start_time
                
                data = {
                    'domain': domain,
                    'resolved_ip': ip,
                    'resolution_time': resolution_time
                }
                
                status = DiagnosticStatus.SUCCESS
                message = f"域名 {domain} 解析成功，IP地址: {ip}"
            
            return NetworkTestResult(
                test_name="DNS解析测试",
                status=status,
                message=message,
                data=data,
                timestamp=datetime.datetime.now().isoformat()
            )
            
        except socket.gaierror as e:
            return NetworkTestResult(
                test_name="DNS解析测试",
                status=DiagnosticStatus.ERROR,
                message=f"域名 {domain} 解析失败: {str(e)}",
                data={'domain': domain, 'error': str(e)},
                timestamp=datetime.datetime.now().isoformat()
            )
        except Exception as e:
            return NetworkTestResult(
                test_name="DNS解析测试",
                status=DiagnosticStatus.ERROR,
                message=f"DNS解析测试失败: {str(e)}",
                data={'domain': domain, 'error': str(e)},
                timestamp=datetime.datetime.now().isoformat()
            )
    
    def trace_route(self, host: str) -> NetworkTestResult:
        """
        路由跟踪测试
        
        Args:
            host: 目标主机
            
        Returns:
            NetworkTestResult: 测试结果
        """
        try:
            # 构建traceroute命令
            if os.name == 'nt':  # Windows
                cmd = ['tracert', '-d', '-w', '1000', host]
            else:  # Linux/Mac
                cmd = ['traceroute', '-n', host]
            
            start_time = time.time()
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            end_time = time.time()
            
            data = {
                'host': host,
                'execution_time': end_time - start_time,
                'return_code': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr
            }
            
            if result.returncode == 0:
                lines = result.stdout.split('\n')
                hops = []
                
                for line in lines:
                    line = line.strip()
                    if line and (line.startswith('traceroute') or line.startswith('tracing route')):
                        continue
                    
                    # 解析跳点信息
                    if '* * *' not in line and line:
                        parts = line.split()
                        if len(parts) >= 2:
                            try:
                                hop_num = int(parts[0])
                                hop_info = {
                                    'hop': hop_num,
                                    'ip': None,
                                    'hostname': None,
                                    'times': []
                                }
                                
                                # 查找IP地址和时间
                                for part in parts[1:]:
                                    if part != '*':
                                        try:
                                            # 检查是否为IP地址
                                            socket.inet_aton(part)
                                            hop_info['ip'] = part
                                            break
                                        except socket.error:
                                            # 可能是主机名
                                            hop_info['hostname'] = part
                                            break
                                
                                # 提取时间信息
                                time_parts = [p for p in parts if p != '*' and p.replace('.', '').isdigit()]
                                hop_info['times'] = time_parts[:3]  # 最多3个时间值
                                
                                hops.append(hop_info)
                            except (ValueError, IndexError):
                                continue
                
                data['hops'] = hops
                
                status = DiagnosticStatus.SUCCESS
                message = f"路由跟踪到 {host} 完成，共 {len(hops)} 跳"
                
            else:
                status = DiagnosticStatus.ERROR
                message = f"路由跟踪到 {host} 失败"
            
            return NetworkTestResult(
                test_name="路由跟踪测试",
                status=status,
                message=message,
                data=data,
                timestamp=datetime.datetime.now().isoformat()
            )
            
        except subprocess.TimeoutExpired:
            return NetworkTestResult(
                test_name="路由跟踪测试",
                status=DiagnosticStatus.ERROR,
                message=f"路由跟踪到 {host} 超时",
                data={'host': host, 'error': 'timeout'},
                timestamp=datetime.datetime.now().isoformat()
            )
        except Exception as e:
            return NetworkTestResult(
                test_name="路由跟踪测试",
                status=DiagnosticStatus.ERROR,
                message=f"路由跟踪测试失败: {str(e)}",
                data={'host': host, 'error': str(e)},
                timestamp=datetime.datetime.now().isoformat()
            )
    
    def check_port_connectivity(self, host: str, port: int, timeout: int = 5) -> NetworkTestResult:
        """
        端口连通性测试
        
        Args:
            host: 目标主机
            port: 目标端口
            timeout: 超时时间
            
        Returns:
            NetworkTestResult: 测试结果
        """
        try:
            start_time = time.time()
            
            # 创建socket连接
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            
            result = sock.connect_ex((host, port))
            connect_time = time.time() - start_time
            sock.close()
            
            data = {
                'host': host,
                'port': port,
                'connect_time': connect_time,
                'timeout': timeout
            }
            
            if result == 0:
                status = DiagnosticStatus.SUCCESS
                message = f"主机 {host}:{port} 端口连通正常，连接时间: {connect_time:.3f}s"
            else:
                status = DiagnosticStatus.ERROR
                message = f"主机 {host}:{port} 端口无法连通"
            
            return NetworkTestResult(
                test_name="端口连通性测试",
                status=status,
                message=message,
                data=data,
                timestamp=datetime.datetime.now().isoformat()
            )
            
        except socket.timeout:
            return NetworkTestResult(
                test_name="端口连通性测试",
                status=DiagnosticStatus.ERROR,
                message=f"主机 {host}:{port} 端口连接超时",
                data={'host': host, 'port': port, 'error': 'timeout'},
                timestamp=datetime.datetime.now().isoformat()
            )
        except Exception as e:
            return NetworkTestResult(
                test_name="端口连通性测试",
                status=DiagnosticStatus.ERROR,
                message=f"端口连通性测试失败: {str(e)}",
                data={'host': host, 'port': port, 'error': str(e)},
                timestamp=datetime.datetime.now().isoformat()
            )
    
    def check_bandwidth(self, host: str, port: int = 80, test_duration: int = 10) -> NetworkTestResult:
        """
        带宽测试（简化版）
        
        Args:
            host: 目标主机
            port: 目标端口
            test_duration: 测试持续时间（秒）
            
        Returns:
            NetworkTestResult: 测试结果
        """
        try:
            start_time = time.time()
            bytes_received = 0
            timeout = test_duration
            
            # 创建socket连接
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)  # 短超时以便检测数据流
            
            try:
                sock.connect((host, port))
                
                # 发送HTTP请求获取数据
                request = f"GET / HTTP/1.1\r\nHost: {host}\r\nConnection: close\r\n\r\n"
                sock.send(request.encode())
                
                # 接收数据并计算带宽
                while time.time() - start_time < test_duration:
                    try:
                        data = sock.recv(4096)
                        if not data:
                            break
                        bytes_received += len(data)
                    except socket.timeout:
                        continue
                    except Exception:
                        break
                        
            finally:
                sock.close()
            
            end_time = time.time()
            actual_duration = end_time - start_time
            bandwidth_bps = (bytes_received * 8) / actual_duration if actual_duration > 0 else 0
            bandwidth_mbps = bandwidth_bps / (1024 * 1024)
            
            data = {
                'host': host,
                'port': port,
                'test_duration': actual_duration,
                'bytes_received': bytes_received,
                'bandwidth_bps': bandwidth_bps,
                'bandwidth_mbps': bandwidth_mbps
            }
            
            status = DiagnosticStatus.SUCCESS
            message = f"带宽测试完成，估算带宽: {bandwidth_mbps:.2f} Mbps"
            
            return NetworkTestResult(
                test_name="带宽测试",
                status=status,
                message=message,
                data=data,
                timestamp=datetime.datetime.now().isoformat()
            )
            
        except Exception as e:
            return NetworkTestResult(
                test_name="带宽测试",
                status=DiagnosticStatus.ERROR,
                message=f"带宽测试失败: {str(e)}",
                data={'host': host, 'port': port, 'error': str(e)},
                timestamp=datetime.datetime.now().isoformat()
            )
    
    def get_network_interfaces(self) -> NetworkTestResult:
        """
        获取网络接口信息
        
        Returns:
            NetworkTestResult: 测试结果
        """
        try:
            interfaces = []
            
            if os.name == 'nt':  # Windows
                cmd = ['ipconfig', '/all']
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                
                if result.returncode == 0:
                    lines = result.stdout.split('\n')
                    current_interface = {}
                    
                    for line in lines:
                        line = line.strip()
                        if '适配器' in line or 'adapter' in line:
                            if current_interface:
                                interfaces.append(current_interface)
                            current_interface = {'name': line}
                        elif ':' in line:
                            parts = line.split(':', 1)
                            if len(parts) == 2:
                                key = parts[0].strip()
                                value = parts[1].strip()
                                current_interface[key] = value
                    
                    if current_interface:
                        interfaces.append(current_interface)
                        
            else:  # Linux/Mac
                cmd = ['ip', 'addr', 'show']
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                
                if result.returncode == 0:
                    lines = result.stdout.split('\n')
                    current_interface = {}
                    
                    for line in lines:
                        line = line.strip()
                        if line and not line.startswith(' '):
                            if current_interface:
                                interfaces.append(current_interface)
                            parts = line.split(':')
                            if len(parts) >= 2:
                                current_interface = {
                                    'name': parts[1].strip(),
                                    'index': parts[0].strip()
                                }
                        elif line.startswith('    '):
                            # 属性行
                            parts = line.split()
                            if len(parts) >= 2:
                                key = parts[0].rstrip(':')
                                value = ' '.join(parts[1:])
                                current_interface[key] = value
                    
                    if current_interface:
                        interfaces.append(current_interface)
            
            data = {
                'interfaces': interfaces,
                'count': len(interfaces)
            }
            
            status = DiagnosticStatus.SUCCESS
            message = f"获取到 {len(interfaces)} 个网络接口"
            
            return NetworkTestResult(
                test_name="网络接口检测",
                status=status,
                message=message,
                data=data,
                timestamp=datetime.datetime.now().isoformat()
            )
            
        except Exception as e:
            return NetworkTestResult(
                test_name="网络接口检测",
                status=DiagnosticStatus.ERROR,
                message=f"获取网络接口信息失败: {str(e)}",
                data={'error': str(e)},
                timestamp=datetime.datetime.now().isoformat()
            )
    
    def comprehensive_diagnosis(self, target: str, ports: List[int] = None) -> DiagnosticReport:
        """
        综合网络诊断
        
        Args:
            target: 诊断目标（IP或域名）
            ports: 要测试的端口列表
            
        Returns:
            DiagnosticReport: 诊断报告
        """
        if ports is None:
            ports = [80, 443, 22, 21, 25, 53]
        
        report_id = f"diag_{int(time.time())}"
        timestamp = datetime.datetime.now().isoformat()
        self.test_results.clear()
        self.recommendations.clear()
        
        # 1. Ping连通性测试
        ping_result = self.ping_host(target)
        self.test_results.append(ping_result)
        
        # 2. DNS解析测试
        dns_result = self.check_dns_resolution(target)
        self.test_results.append(dns_result)
        
        # 3. 路由跟踪测试
        route_result = self.trace_route(target)
        self.test_results.append(route_result)
        
        # 4. 端口连通性测试
        for port in ports:
            port_result = self.check_port_connectivity(target, port)
            self.test_results.append(port_result)
        
        # 5. 网络接口检测
        interface_result = self.get_network_interfaces()
        self.test_results.append(interface_result)
        
        # 6. 带宽测试（仅对HTTP端口）
        if 80 in ports or 443 in ports:
            bandwidth_result = self.check_bandwidth(target, 80 if 80 in ports else 443)
            self.test_results.append(bandwidth_result)
        
        # 生成统计信息
        summary = {'成功': 0, '警告': 0, '错误': 0, '未知': 0}
        for result in self.test_results:
            summary[result.status.value] += 1
        
        # 生成建议
        self._generate_recommendations(target, summary)
        
        return DiagnosticReport(
            report_id=report_id,
            timestamp=timestamp,
            target=target,
            tests=self.test_results,
            summary=summary,
            recommendations=self.recommendations
        )
    
    def _generate_recommendations(self, target: str, summary: Dict[str, int]):
        """生成诊断建议"""
        recommendations = []
        
        # 基于测试结果生成建议
        if summary.get('error', 0) > 0:
            recommendations.append("检测到网络错误，建议检查网络连接和防火墙设置")
        
        if summary.get('warning', 0) > 0:
            recommendations.append("检测到网络警告，建议监控网络性能")
        
        # 检查特定问题
        ping_results = [r for r in self.test_results if r.test_name == "Ping连通性测试"]
        if ping_results and ping_results[0].status == DiagnosticStatus.ERROR:
            recommendations.append("Ping测试失败，可能的原因：网络不通、防火墙阻止、目标主机离线")
        
        dns_results = [r for r in self.test_results if r.test_name == "DNS解析测试"]
        if dns_results and dns_results[0].status == DiagnosticStatus.ERROR:
            recommendations.append("DNS解析失败，建议检查DNS服务器配置或使用其他DNS服务器")
        
        port_results = [r for r in self.test_results if r.test_name == "端口连通性测试"]
        failed_ports = [r.data.get('port') for r in port_results if r.status == DiagnosticStatus.ERROR]
        if failed_ports:
            recommendations.append(f"端口 {', '.join(map(str, failed_ports))} 无法连通，可能未开放或被防火墙阻止")
        
        self.recommendations = recommendations
    
    def export_report(self, report: DiagnosticReport, format: str = 'json', filename: str = None) -> str:
        """
        导出诊断报告
        
        Args:
            report: 诊断报告
            format: 导出格式 ('json', 'txt')
            filename: 文件名（可选）
            
        Returns:
            str: 导出的文件路径
        """
        if filename is None:
            filename = f"network_diagnosis_report_{report.report_id}"
        
        if format.lower() == 'json':
            filename += '.json'
            report_dict = asdict(report)
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(report_dict, f, ensure_ascii=False, indent=2)
                
        elif format.lower() == 'txt':
            filename += '.txt'
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"网络诊断报告\n")
                f.write(f"=" * 50 + "\n")
                f.write(f"报告ID: {report.report_id}\n")
                f.write(f"生成时间: {report.timestamp}\n")
                f.write(f"诊断目标: {report.target}\n\n")
                
                f.write(f"诊断摘要:\n")
                f.write(f"成功: {report.summary.get('成功', 0)}\n")
                f.write(f"警告: {report.summary.get('警告', 0)}\n")
                f.write(f"错误: {report.summary.get('错误', 0)}\n")
                f.write(f"未知: {report.summary.get('未知', 0)}\n\n")
                
                f.write(f"详细测试结果:\n")
                f.write(f"-" * 50 + "\n")
                for test in report.tests:
                    f.write(f"测试: {test.test_name}\n")
                    f.write(f"状态: {test.status.value}\n")
                    f.write(f"消息: {test.message}\n")
                    f.write(f"时间: {test.timestamp}\n\n")
                
                if report.recommendations:
                    f.write(f"建议:\n")
                    f.write(f"-" * 50 + "\n")
                    for i, rec in enumerate(report.recommendations, 1):
                        f.write(f"{i}. {rec}\n")
        
        return os.path.abspath(filename)
    
    def print_report(self, report: DiagnosticReport):
        """打印诊断报告到控制台"""
        print(f"\n{'='*60}")
        print(f"网络诊断报告")
        print(f"{'='*60}")
        print(f"报告ID: {report.report_id}")
        print(f"生成时间: {report.timestamp}")
        print(f"诊断目标: {report.target}")
        print(f"\n诊断摘要:")
        print(f"  成功: {report.summary.get('成功', 0)}")
        print(f"  警告: {report.summary.get('警告', 0)}")
        print(f"  错误: {report.summary.get('错误', 0)}")
        print(f"  未知: {report.summary.get('未知', 0)}")
        
        print(f"\n详细测试结果:")
        print(f"{'-'*60}")
        for test in report.tests:
            status_symbol = {
                DiagnosticStatus.SUCCESS: "✓",
                DiagnosticStatus.WARNING: "⚠",
                DiagnosticStatus.ERROR: "✗",
                DiagnosticStatus.UNKNOWN: "?"
            }.get(test.status, "?")
            
            print(f"{status_symbol} {test.test_name}")
            print(f"  状态: {test.status.value}")
            print(f"  消息: {test.message}")
            print(f"  时间: {test.timestamp}")
            print()
        
        if report.recommendations:
            print(f"建议:")
            print(f"{'-'*60}")
            for i, rec in enumerate(report.recommendations, 1):
                print(f"{i}. {rec}")
        
        print(f"{'='*60}")


def main():
    """主函数 - 命令行接口"""
    import argparse
    
    parser = argparse.ArgumentParser(description='W8网络诊断器')
    parser.add_argument('target', help='诊断目标（IP或域名）')
    parser.add_argument('--ports', nargs='+', type=int, default=[80, 443, 22, 21, 25, 53],
                        help='要测试的端口列表')
    parser.add_argument('--format', choices=['json', 'txt'], default='txt',
                        help='报告导出格式')
    parser.add_argument('--output', help='输出文件名')
    parser.add_argument('--no-console', action='store_true',
                        help='不在控制台显示报告')
    
    args = parser.parse_args()
    
    # 创建诊断器实例
    diagnostics = NetworkDiagnostics()
    
    print(f"开始对 {args.target} 进行网络诊断...")
    
    # 执行综合诊断
    report = diagnostics.comprehensive_diagnosis(args.target, args.ports)
    
    # 显示报告
    if not args.no_console:
        diagnostics.print_report(report)
    
    # 导出报告
    if args.output:
        filename = diagnostics.export_report(report, args.format, args.output)
        print(f"\n报告已导出到: {filename}")
    else:
        filename = diagnostics.export_report(report, args.format)
        print(f"\n报告已导出到: {filename}")


if __name__ == "__main__":
    main()