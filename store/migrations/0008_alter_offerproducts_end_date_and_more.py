# Generated by Django 4.2.2 on 2023-07-22 12:50

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('store', '0007_alter_offerproducts_is_active'),
    ]

    operations = [
        migrations.AlterField(
            model_name='offerproducts',
            name='end_date',
            field=models.DateTimeField(auto_now=True),
        ),
        migrations.AlterField(
            model_name='offerproducts',
            name='start_date',
            field=models.DateTimeField(auto_now_add=True),
        ),
    ]
