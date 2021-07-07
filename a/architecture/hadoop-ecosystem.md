# hadoop ecosystem

OLTP -&gt; OLAP \(cube, hierarchy\)

1.基于hbase预聚合的，比如Opentsdb, Kylin, Druid等,需要指定预聚合的指标，在数据接入的时候根据指定的指标进行聚合运算，适合相对固定的业务报表类需求，只需要统计少量维度即可满足业务报表需求

2.基于Parquet列式存储的，比如Presto, Drill, Impala等，基本是完全基于内存的并行计算，Parquet系能降低存储空间，提高IO效率，以离线处理为主，很难提高数据写的实时性，超大表的join支持可能不够好。spark sql也算类似，但它在内存不足时可以spill disk来支持超大数据查询和join

基本上 Hive 能读的数据 Presto 也都能读

hive&hbase: hive依赖hadoop，而hbase仅依赖hadoop的hdfs模块

zookeeper \(HA高可用，各组件主备选举管理\)

yarn 

sqoop \(关系型数据库&lt;-&gt;hadoop存储\)

oozie \(workflow\)

hue \(interface of all\)

knox \(on Jetty, webui重定向,访问管理\)

flume \(非侵入日志采集和消费\)

storm - Jstorm -\|

flink

ganglia + nagios = ambari \(集群机器监控报警\)

tez \(优化的mapred, java/hive..\)

mapreduce -&gt; spark

