import pandas as pd
import os

path = "H:/final_data/DOTA2_dataset/raw_dataset"
files = os.listdir(path)

global zhui
zhui = 1

for file in files:
    position = path + '\\' + file
    data1 = pd.read_csv(position, header=None)
    data2 = data1[0].str.split(";", expand=True)

    time = pd.Series([30 * x for x in range(0, 120)] * 14).sort_values().reset_index(drop=True)

    towers = data1.loc[data1.index % 14 == 2].apply(
        lambda x: [int(x[0].split(":")[i].split("/")[0]) != 0 for i in range(1, 19)], axis=1)
    tower2_count = towers.apply(lambda x: sum(x[:9]))
    tower3_count = towers.apply(lambda x: sum(x[9:]))

    tower = pd.concat([pd.Series(tower2_count).repeat(9), pd.Series(tower3_count).repeat(5)]).reset_index(drop=True)

    forts = data1.loc[data1.index % 14 == 1].apply(
        lambda x: [int(x[0].split(";")[i].split(":")[1]) for i in range(0, 2)], axis=1)
    fort2_count = forts.apply(lambda x: x[0])
    fort3_count = forts.apply(lambda x: x[1])

    fort = pd.concat([pd.Series(fort2_count).repeat(9), pd.Series(fort3_count).repeat(5)]).reset_index(drop=True)

    barracks = data1.loc[data1.index % 14 == 2].apply(
        lambda x: [int(x[0].split(":")[i].split("/")[0]) != 0 for i in range(1, 13)], axis=1)
    barracks2_count = barracks.apply(lambda x: sum(x[:6]))
    barracks3_count = barracks.apply(lambda x: sum(x[6:]))

    barracks = pd.concat([pd.Series(barracks2_count).repeat(9), pd.Series(barracks3_count).repeat(5)]).reset_index(
        drop=True)

    hero = data2[0].str.split("_", expand=True)[3]
    ID = data2[1].str.split(":", expand=True)[1]
    strength = data2[2].str.split(":", expand=True)[1]
    agility = data2[3].str.split(":", expand=True)[1]
    intellect = data2[4].str.split(":", expand=True)[1]
    health = data2[5].str.split(":", expand=True)[1].str.split("/", expand=True)[0]
    mana = data2[6].str.split(":", expand=True)[1].str.split("/", expand=True)[0]
    current_level = data2[7].str.split(":", expand=True)[1]
    current_XP = data2[8].str.split(":", expand=True)[1]
    damage_min = data2[9].str.split(":", expand=True)[1]
    damage_max = data2[10].str.split(":", expand=True)[1]
    damage_bonus = data2[11].str.split(":", expand=True)[1]
    move_speed = data2[12].str.split(":", expand=True)[1]
    team = data2[13].str.split(":", expand=True)[1]
    gold = data2[14].str.split(":", expand=True)[1]
    LH = data2[15].str.split(":", expand=True)[1]
    DN = data2[16].str.split(":", expand=True)[1]
    kill = data2[17].str.split(":", expand=True)[1]
    death = data2[18].str.split(":", expand=True)[1]
    assist = data2[19].str.split(":", expand=True)[1]

    win = data1.iloc[len(data1) - 1][0][7::]
    winner = pd.Series([win] * len(data1))

    data = pd.DataFrame({"time": time, "hero": hero, "ID": ID, "team": team, "strength": strength, "agility": agility,
                         "intellect": intellect, "health": health, "mana": mana, "current_level": current_level,
                         "current_XP": current_XP, "damage_max": damage_max, "damage_bonus": damage_bonus,
                         "move_speed": move_speed, "gold": gold, "LH": LH, "DH": DN, "kill": kill, "death": death,
                         "assist": assist, "rest_tower": tower, "rest_barracks": barracks, "fort": fort,
                         "winner": winner})

    Data = data[data['hero'].notnull()]
    Data.to_csv("H:/final_data/chulidata/data" + str(zhui) + ".csv", index=False, sep=',')
    zhui += 1

