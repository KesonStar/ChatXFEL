from pymilvus import connections, utility, Collection

connections.connect(
    alias="default",
    host="10.19.48.181",
    port="19530",
    user="cs286_2025_group8",     # 你的账号
    password="Group8",              # 你的密码
    db_name="cs286_2025_group8",    # ⭐ 必须指定你在 UI 中看到的数据库名
    secure=False,                   # 内网一般不启用 TLS
)


col = Collection("xfel_bibs_collection")
print(col.schema)


print("\n=== Fields ===")
fields = [f.name for f in col.schema.fields]
print(fields)

# 选择你想看的字段（向量先不展示）
output_fields = ["pk", "title", "year", "journal", "page", "text", "start_index"]

print("\n=== First 3 Rows ===")
res = col.query(
    expr="pk >=0 ",
    output_fields=output_fields,
    limit=3
)

for r in res:
    print("\n--- Row ---")
    for k, v in r.items():
        print(f"{k}: {v}")